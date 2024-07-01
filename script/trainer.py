import torch
from torch import nn, optim
import time
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils import logger
from script import evaluation
from accelerate import Accelerator
import math
from .trainer_utils import *
from transformers.trainer import ALL_LAYERNORM_LAYERS


class Trainer:
    def __init__(self, config, model, data) -> None:
        self.config = config
        self.model = model
        self.data = data

        self.writer = SummaryWriter(config.tensorboard_log_dir)
        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)
        self.init_param()

    def train(self):
        train_dataloader = self.data.train_dataloader()
        optimizer, scheduler = self.init_optimizer()
        global_step = 0
        log_step = 0
        self.best_f1 = 0

        optimizer, train_dataloader = self.accelerator.prepare(optimizer, train_dataloader)

        self.model.log_model_weights()
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            self.model.train()
            logger.info('Epoch: ' + str(epoch))

            overall_arc_correct, overall_arc_total, overall_rel_correct = 0, 0, 0
            for batch, onebatch in enumerate(train_dataloader, start=1):
                with self.accelerator.autocast():
                    _, inputs, gold_labels = onebatch
                    self.model.forward(inputs)

                    loss = self.model.compute_loss(**gold_labels)

                    loss = loss / self.config.update_every
                    self.accelerator.backward(loss)
                    loss_value = loss.data.item()

                    arc_correct, arc_total, rel_correct = self.model.compute_accuracy(**gold_labels)

                    overall_rel_correct += rel_correct
                    overall_arc_correct += arc_correct
                    overall_arc_total += arc_total

                    uas = overall_arc_correct * 100.0 / overall_arc_total
                    las = overall_rel_correct * 100.0 / overall_arc_total
                    during_time = float(time.time() - start_time)

                    self.writer.add_scalar("train/loss", loss_value, log_step)
                    self.writer.add_scalar("train/uas", uas, log_step)
                    self.writer.add_scalar("train/las", las, log_step)
                    log_step += 1
                    logger.info(
                        f"Step: {global_step:03}, Epoch: {epoch:03}, batch: {batch:03}, time: {during_time:.2f}, uas: {uas:.2f}, las: {las:.2f}, loss, {loss_value:.2f}"
                    )

                    if batch % self.config.update_every == 0 or batch == len(train_dataloader):
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.clip)  # 梯度剪裁
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                    self.writer.add_scalar("parameters/plm_lr", scheduler.get_lr()[0], log_step)
                    self.writer.add_scalar("parameters/inter_lr", scheduler.get_lr()[1], log_step)
                    self.writer.add_scalar("parameters/other_lr", scheduler.get_lr()[2], log_step)

                    if batch % self.config.validate_every == 0 or batch == len(train_dataloader):
                        self.evaluation(global_step)
                        self.update_steps += 1

            if self.config.early_stop and self.update_steps > self.config.early_stop and batch == len(train_dataloader):
                logger.info("#" * 30)
                logger.info(f"Not updated for more than {self.update_steps} rounds, stop early.")
                logger.info("#" * 30)
                break

        if self.config.best_model_path:
            logger.info("*" * 10 + f"Load best model from {self.config.best_model_path}" + "*" * 10)
            self.model.load_state_dict(torch.load(self.config.best_model_path), strict=True)
            self.evaluation(global_step + 100, True)  # 区分最大值用的

    def log_evaluation(self, dev_metrics, test_metrics, eval_loss, global_step, best):
        """-----------------DEV-----------------"""
        dev_uas_metric, dev_las_metric = dev_metrics
        dev_uas_f1 = dev_uas_metric["f1_score"] * 100
        dev_las_f1 = dev_las_metric["f1_score"] * 100
        logger.info(f"DEV:")
        logger.info(f"UAS F1 score: {dev_uas_f1: 5.4f}, LAS F1 score: {dev_las_f1: 5.4f},  Loss: {eval_loss: 5.2f}")

        self.writer.add_scalar("dev/uas f1 score", dev_uas_f1, global_step)
        self.writer.add_scalar("dev/las f1 score", dev_las_f1, global_step)
        self.writer.add_scalar("dev/eval_loss", eval_loss, global_step)

        """-----------------TEST-----------------"""
        if best or test_metrics:
            test_uas_metric, test_las_metric = test_metrics
            test_uas_f1 = test_uas_metric["f1_score"] * 100
            test_las_f1 = test_las_metric["f1_score"] * 100
            logger.info(f"TEST:")
            logger.info(f"UAS F1 score: {test_uas_f1: 5.4f}, LAS F1 score: {test_las_f1: 5.4f}")

            self.writer.add_scalar("test/uas f1 score", test_uas_f1, global_step)
            self.writer.add_scalar("test/las f1 score", test_las_f1, global_step)

        if best:
            logger.info(f"{dev_uas_f1:5.2f} {dev_las_f1:5.2f} {test_uas_f1:5.2f} {test_las_f1:5.2f}")

    def evaluation(self, global_step, best=False):
        logger.info("-" * 42 + "  Dev and Test  " + "-" * 42)

        with torch.no_grad():
            predict_dev_file = self.config.dev_file + "." + str(global_step)
            dev_metrics, eval_loss = self.predict(self.config.dev_file, predict_dev_file, stage="eval")

            if best:
                predict_test_file = self.config.test_file + "." + str(global_step)
                test_meterics, _ = self.predict(self.config.test_file, predict_test_file, stage="test")
            else:
                test_meterics = None

            self.log_evaluation(dev_metrics, test_meterics, eval_loss, global_step, best)

            monitor = dev_metrics[1]["f1_score"]
            if monitor > self.best_f1:
                logger.info(f"Exceed best Full F-score: history = {self.best_f1}, current = {monitor:.2f}")
                self.best_f1 = dev_metrics[1]["f1_score"]
                torch.save(self.model.state_dict(), self.config.save_model_path)
                logger.info(f'Saving model to {self.config.save_model_path}')
                self.config.best_model_path = self.config.save_model_path

                self.update_steps = 0

        logger.info("-" * 100)

    def predict(self, gold_file, out_file, stage="eval", ckpt_path=None):
        start_time = time.time()
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

        if stage == "eval":
            dataloader = self.data.val_dataloader()
        elif stage == "test":
            dataloader = self.data.test_dataloader()
        else:
            raise KeyError("""stage must in ["test", "eval"]""")

        dataloader = self.accelerator.prepare(dataloader)
        instances = []
        all_losses = []
        for onebatch in dataloader:
            batch_instances, inputs, gold_labels = onebatch
            self.model.forward(inputs)
            loss = self.model.compute_loss(**gold_labels)
            all_losses.append(loss.data.item())

            pred_arcs, pred_rels = self.model.decode(inputs["edu_lengths"])
            instances.extend(batch_instances)

            for batch_index, (arcs, rels) in enumerate(zip(pred_arcs, pred_rels)):
                instance = batch_instances[batch_index]
                length = len(instance.edus)
                relation_list = instance.pred_relations.copy()
                for idx in range(length):
                    if idx == 0 or relation_list[idx - 1]["x"] == -1:
                        continue
                    y = idx
                    x = int(arcs[idx])
                    type = self.data.vocab.id2rel(rels[idx])
                    relation = dict()
                    relation['y'] = y
                    relation['x'] = x
                    relation['type'] = type
                    relation_list[idx - 1] = relation
                instance.pred_relations = relation_list

        self.data.write_instances(out_file, instances)
        self.model.train()

        during_time = time.time() - start_time
        logger.info(f"Doc num: {len(instances)},  parser time: {during_time:.2f}")

        loss = sum(all_losses) / len(all_losses)
        return evaluation(gold_file, out_file), loss

    def get_group_parameter(self, model):
        total_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in total_parameters if "bias" not in name]

        plm_parameters = []
        inter_parameters = []
        fusion_parameters = []
        other_parameters = []
        for name in total_parameters:
            if "plm" in name:
                plm_parameters.append(name)
            elif "inter_module" in name:
                inter_parameters.append(name)
            elif "fusion_t" in name:
                fusion_parameters.append(name)
            else:
                other_parameters.append(name)

        optimizer_grouped_parameters = [
            {
                # plm参数 且要decay
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and n in plm_parameters)],
                "weight_decay": self.config.l2_reg,
                "lr": self.config.plm_learning_rate,
            },
            {
                # plm外的参数 且要decay
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and n in other_parameters)],
                "weight_decay": self.config.l2_reg,
                "lr": self.config.learning_rate,
            },
            {
                # att参数 且要decay
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and n in inter_parameters)],
                "weight_decay": self.config.l2_reg,
                "lr": self.config.inter_learning_rate,
            },
            {
                # fusion的参数 且要decay
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and n in fusion_parameters)],
                "weight_decay": self.config.l2_reg,
                "lr": self.config.fusion_lr,
            },
            # 以上需要decay 以下不需要
            {
                # plm的参数 且不要decay
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in plm_parameters)],
                "weight_decay": 0.0,
                "lr": self.config.plm_learning_rate,
            },
            {
                # plm外的参数 且不要decay
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in other_parameters)],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate,
            },
            {
                # att的参数 且不要decay
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in inter_parameters)],
                "weight_decay": 0.0,
                "lr": self.config.inter_learning_rate,
            },
            {
                # fusion的参数 且不要decay
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in fusion_parameters)],
                "weight_decay": 0.0,
                "lr": self.config.fusion_lr,
            },
        ]

        return optimizer_grouped_parameters

    def optimizer_grouped_parameters(self, weight_decay):
        """
        为模型参数设置不同的优化器超参 比如分层学习率等
        """
        total_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in total_parameters if "bias" not in name]

        plm_lr_parameters = []
        inter_lr_parameters = []
        for name in total_parameters:
            if 'plm' in name:
                plm_lr_parameters.append(name)

            if "inter_module" in name:
                inter_lr_parameters.append(name)
        optimizer_grouped_parameters = [
            {
                # plm参数 且要decay
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n in plm_lr_parameters],
                "weight_decay": weight_decay,
                "lr": self.config.plm_learning_rate,
            },
            {
                # plm外的参数 且要decay
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters and n not in plm_lr_parameters and n not in inter_lr_parameters
                ],
                "weight_decay": weight_decay,
                "lr": self.config.learning_rate,
            },
            {
                # att参数 且要decay
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n in inter_lr_parameters],
                "weight_decay": weight_decay,
                "lr": self.config.inter_learning_rate,
            },
            {
                # plm的参数 且不要decay
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n in plm_lr_parameters],
                "weight_decay": 0.0,
                "lr": self.config.plm_learning_rate,
            },
            {
                # plm外的参数 且不要decay
                "params": [
                    p for n, p in self.model.named_parameters() if n not in decay_parameters and n not in plm_lr_parameters and n not in inter_lr_parameters
                ],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate,
            },
            {
                # att的参数 且不要decay
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n in inter_lr_parameters],
                "weight_decay": 0.0,
                "lr": self.config.inter_learning_rate,
            },
        ]

        return optimizer_grouped_parameters

    def init_optimizer(self):
        # model_param = self.optimizer_grouped_parameters(self.config.l2_reg)
        model_param = self.get_group_parameter(self.model)
        optim = torch.optim.AdamW(
            model_param, lr=self.config.learning_rate, betas=(self.config.beta_1, self.config.beta_2), eps=self.config.epsilon, weight_decay=self.config.l2_reg
        )

        decay, decay_step = self.config.decay, self.config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
        #     scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        # )

        return optim, scheduler

    def init_param(self):
        if self.config.val_check_interval:
            self.config.validate_every = math.ceil((len(self.data.train_dataloader()) / self.config.train_batch_size) * self.config.val_check_interval)

        self.config.best_model_path = self.config.ckpt_path

        self.update_steps = 0  # 记录上一次更新的次数
