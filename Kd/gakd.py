import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import json
import time
from tqdm import tqdm
import argparse
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from baseline import GINENetwork
import pandas as pd
from datetime import datetime

base_dir = f"/scratch1/users/u12763/Knowledge-distillation/"
if not os.path.exists(base_dir):
    os.makedirs(base_dir, exist_ok=True)


class logits_D(nn.Module):
    def __init__(self, n_class, n_hidden):
        super(logits_D, self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.lin = nn.Linear(self.n_class, self.n_hidden)  # assert n_class==n_hidden
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.n_hidden, self.n_class + 1, bias=False)

    def forward(self, logits, temperature=1.0):
        out = self.lin(logits / temperature)
        out = logits + out
        out = self.relu(out)
        dist = self.lin2(out)
        return dist


class local_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(local_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden,)))
        self.scale = nn.Parameter(torch.full(size=(1,), fill_value=1.0))

    def forward(self, emb, batch):
        emb = F.normalize(emb, p=2)
        (u, v) = batch.edge_index
        euw = emb[u] @ torch.diag(self.d)
        pair_dis = euw @ emb[v].t()
        return torch.diag(pair_dis) * self.scale


class global_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(global_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden,)))
        self.scale = nn.Parameter(torch.full(size=(1,), fill_value=1.0))

    def forward(self, emb, summary, batch):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d)
        sims = []
        for i, s in enumerate(summary):
            pre, post = batch.ptr[i], batch.ptr[i + 1]
            sims.append(sim[pre:post] @ s.unsqueeze(-1))
        sim = torch.cat(sims, dim=0).squeeze(-1)
        return sim * self.scale


def load_knowledge(kd_path, device):  # load teacher knowledge
    assert os.path.isfile(kd_path), "Please download teacher knowledge first"
    knowledge = torch.load(kd_path, map_location=device)
    tea_logits = knowledge["logits"].float()
    tea_h = knowledge["h-embedding"]
    tea_g = knowledge["g-embedding"]
    new_ptr = knowledge["ptr"]
    return tea_logits, tea_h, tea_g, new_ptr


class GAKD_trainer:

    def __init__(
        self,
        student_model: nn.Module,
        teacher_knowledge_path: str,
        dataset_name="ogbg-molpcba",
        embedding_dim=400,
        student_lr=5e-3,
        student_weight_decay=1e-5,
        discriminator_lr=1e-2,
        discriminator_weight_decay=5e-4,
        batch_size=32,
        num_workers=4,
        discriminator_update_freq=5,  # K in paper
        epochs=100,
        seed=42,
    ):
        self.seed = seed
        self.dataset_name = dataset_name
        self.student_model = student_model
        self.teacher_knowledge_path = teacher_knowledge_path
        self.embedding_dim = embedding_dim
        self.student_lr = student_lr
        self.student_weight_decay = student_weight_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_weight_decay = discriminator_weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.discriminator_update_freq = discriminator_update_freq
        self.setup()

    def setup(self):
        self._set_device()
        self._set_seed(self.seed)
        self._load_knowledge()
        self._load_dataset()
        self._setup_student()
        self._setup_discriminator()
        self.evaluate_teacher()

    def _load_dataset(self):
        os.makedirs(f"{base_dir}/data", exist_ok=True)
        self.dataset = PygGraphPropPredDataset(
            name=self.dataset_name, root=f"{base_dir}/data"
        )
        self.split_idx = self.dataset.get_idx_split()

        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.dataset[self.split_idx["train"]],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # pin_memory=True,  # Enables faster data transfer to GPU
            # persistent_workers=True  # Keeps workers alive between epochs
        )
        self.valid_loader = DataLoader(
            self.dataset[self.split_idx["valid"]],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            self.dataset[self.split_idx["test"]],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.evaluator = Evaluator(name=self.dataset_name)

    def _setup_student(self):
        self.student_model = self.student_model.to(self.device)
        self.student_optimizer = optim.Adam(
            self.student_model.parameters(),
            lr=self.student_lr,
            weight_decay=self.student_weight_decay,
        )

    def _setup_discriminator(self):
        self.discriminator_e_local = local_emb_D(n_hidden=self.embedding_dim).to(
            self.device
        )
        self.discriminator_e_global = global_emb_D(n_hidden=self.embedding_dim).to(
            self.device
        )
        self.discriminator_logits = logits_D(
            n_class=self.dataset.num_tasks, n_hidden=self.dataset.num_tasks
        ).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            [
                {
                    "params": self.discriminator_e_local.parameters(),
                    "lr": self.discriminator_lr,
                    "weight_decay": self.discriminator_weight_decay,
                },
                {
                    "params": self.discriminator_e_global.parameters(),
                    "lr": self.discriminator_lr,
                    "weight_decay": self.discriminator_weight_decay,
                },
                {
                    "params": self.discriminator_logits.parameters(),
                    "lr": self.discriminator_lr,
                    "weight_decay": self.discriminator_weight_decay,
                },
            ],
            lr=self.discriminator_lr,
            weight_decay=self.discriminator_weight_decay,
        )
        self.discriminator_loss = torch.nn.BCELoss()
        self.class_criterion = torch.nn.BCEWithLogitsLoss()
        self._trains_ids = self.split_idx["train"].to(self.device)

    def _set_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS is currently slower than CPU due to missing int64 min/max ops
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}", flush=True)
        self.device = device

    def _load_knowledge(self, teacher_model_path):
        assert os.path.isfile(
            self.teacher_knowledge_path
        ), "Please download teacher knowledge first"
        knowledge = torch.load(self.teacher_knowledge_path, map_location=self.device)
        self.teacher_logits = knowledge["logits"].float().to(self.device)
        self.teacher_h = knowledge["h-embedding"].to(self.device)
        self.teacher_g = knowledge["g-embedding"].to(self.device)
        self.teacher_ptr = knowledge["ptr"].to(self.device)

    def evaluate_teacher(self):
        train_y_true = self.dataset[self.split_idx["train"]].y
        train_y_pred = self.teacher_logits
        input_dict = {"y_true": train_y_true, "y_pred": train_y_pred}
        if self.dataset_name == "ogbg-molpcba":
            print(
                f"Teacher performance on Training set: {self.evaluator.eval(input_dict)['ap']}",
                flush=True,
            )
        else:
            print(
                f"Teacher performance on Training set: {self.evaluator.eval(input_dict)['rocauc']}",
                flush=True,
            )

    def _set_seed(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}", flush=True)

    def _get_batch_idx_from_teacher(self, batch):
        new_pre = self.teacher_ptr[:-1]
        new_post = self.teacher_ptr[1:]
        new_ids = [(self._trains_ids == vid).nonzero().item() for vid in batch.id]
        batch_graph_idx = torch.tensor(new_ids, device=self.device)
        batch_pre = new_pre[batch_graph_idx]
        batch_post = new_post[batch_graph_idx]
        batch_node_idx = torch.cat(
            [
                torch.arange(pre, post)
                for pre, post in list(zip(*[batch_pre, batch_post]))
            ],
            dim=0,
        )
        return batch_graph_idx, batch_node_idx

    def _train_batch(self, batch, epoch):
        batch = batch.to(self.device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            return

        student_batch_pred, student_batch_h, student_batch_g = self.student_model(batch)
        self.student_optimizer.zero_grad()
        y_true = batch.y.float()
        y_labeled = ~torch.isnan(y_true)

        # classification loss ===> to be used for logits identifier in training student
        class_loss = self.class_criterion(
            student_batch_pred.float()[y_labeled], y_true[y_labeled]
        )

        batch_graph_idx, batch_node_idx = self._get_batch_idx_from_teacher(batch)
        teacher_batch_h = self.teacher_h[batch_node_idx].to(self.device)
        teacher_batch_g = self.teacher_g[batch_graph_idx].to(self.device)
        teacher_batch_logits = self.teacher_logits[batch_graph_idx].to(self.device)

        #### Train discriminator
        if epoch % self.discriminator_update_freq == 0:
            discriminator_loss = 0

            ## train logits identifier: D_l
            self.discriminator_logits.train()
            # detach student logits to avoid backprop through student for discriminator training
            student_logits = student_batch_pred.detach()
            z_teacher = self.discriminator_logits(teacher_batch_logits)
            z_student = self.discriminator_logits(student_logits)
            prob_real_given_z = torch.sigmoid(z_teacher[:, -1])
            prob_fake_given_z = torch.sigmoid(z_student[:, -1])
            adversarial_logits_loss = self.discriminator_loss(
                prob_real_given_z, torch.ones_like(prob_real_given_z)
            ) + self.discriminator_loss(
                prob_fake_given_z, torch.zeros_like(prob_fake_given_z)
            )
            y_v_given_z_pos = self.class_criterion(
                z_teacher[:, -1][y_labeled], y_true[y_labeled]
            )
            y_v_given_z_neg = self.class_criterion(
                z_student[:, -1][y_labeled], y_true[y_labeled]
            )
            label_loss = y_v_given_z_pos + y_v_given_z_neg
            discriminator_loss = 0.5 * (adversarial_logits_loss + label_loss)

            ## train local embedding representation identifier: D_e_local
            self.discriminator_e_local.train()
            pos_e = self.discriminator_e_local(teacher_batch_h, batch)
            neg_e = self.discriminator_e_local(student_batch_h, batch)
            prob_real_given_e = torch.sigmoid(pos_e)
            prob_fake_given_e = torch.sigmoid(neg_e)
            adverserial_local_e_loss = self.discriminator_loss(
                prob_real_given_e, torch.ones_like(prob_real_given_e)
            ) + self.discriminator_loss(
                prob_fake_given_e, torch.zeros_like(prob_fake_given_e)
            )

            ## train global embedding representation identifier: D_e_global
            self.discriminator_e_global.train()
            teacher_summary = torch.sigmoid(teacher_batch_g)
            e_teacher_summary_teacher = self.discriminator_e_global(
                teacher_batch_h, teacher_summary, batch
            )
            e_student_summary_teacher = self.discriminator_e_global(
                student_batch_h.detach(), teacher_summary, batch
            )
            prob_real_given_e_global = torch.sigmoid(e_teacher_summary_teacher)
            prob_fake_given_e_global = torch.sigmoid(e_student_summary_teacher)
            adverserial_global_e_loss1 = self.discriminator_loss(
                prob_real_given_e_global,
                torch.ones_like(prob_real_given_e_global),
            ) + self.discriminator_loss(
                prob_fake_given_e_global,
                torch.zeros_like(prob_fake_given_e_global),
            )

            student_summary = torch.sigmoid(student_batch_g)
            e_student_summary_student = self.discriminator_e_global(
                student_batch_h.detach(), student_summary.detach(), batch
            )
            e_teacher_summary_student = self.discriminator_e_global(
                teacher_batch_h, student_summary.detach(), batch
            )
            prob_real_given_e_global = torch.sigmoid(e_student_summary_student)
            prob_fake_given_e_global = torch.sigmoid(e_teacher_summary_student)
            adverserial_global_e_loss2 = self.discriminator_loss(
                prob_real_given_e_global,
                torch.ones_like(prob_real_given_e_global),
            ) + self.discriminator_loss(
                prob_fake_given_e_global,
                torch.ones_like(prob_fake_given_e_global),
            )

            discriminator_loss = (
                discriminator_loss
                + adverserial_local_e_loss
                + adverserial_global_e_loss1
                + adverserial_global_e_loss2
            )
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        #### Train student
        student_loss = class_loss

        ## fooling logits discriminator
        self.discriminator_logits.eval()
        z_teacher = self.discriminator_logits(teacher_batch_logits)
        z_student = self.discriminator_logits(student_batch_pred)
        prob_fake_given_z = torch.sigmoid(z_student[:, -1])
        adversarial_logits_loss = self.discriminator_loss(
            prob_fake_given_z, torch.ones_like(prob_fake_given_z)
        )
        label_loss = self.class_criterion(
            z_student[:, -1][y_labeled], y_true[y_labeled]
        )
        l1_loss = (
            torch.norm(student_batch_pred - teacher_batch_logits, p=1)
            * 1
            / len(batch.id)
        )
        student_loss = (
            student_loss + 0.5 * (adversarial_logits_loss + label_loss) + l1_loss
        )

        ## fooling local embedding representation identifier
        self.discriminator_e_local.eval()
        neg_e = self.discriminator_e_local(student_batch_h, batch)
        prob_fake_given_e = torch.sigmoid(neg_e)
        adversarial_local_e_loss = self.discriminator_loss(
            prob_fake_given_e, torch.ones_like(prob_fake_given_e)
        )

        ## fooling global embedding representation identifier
        self.discriminator_e_global.eval()
        teacher_summary = torch.sigmoid(teacher_batch_g)
        e_student_summary_teacher = self.discriminator_e_global(
            student_batch_h, teacher_summary, batch
        )
        prob_fake_given_e_global = torch.sigmoid(e_student_summary_teacher)
        adverserial_global_e_loss1 = self.discriminator_loss(
            prob_fake_given_e_global, torch.ones_like(prob_fake_given_e_global)
        )

        student_summary = torch.sigmoid(student_batch_g)
        e_teacher_summary_student = self.discriminator_e_global(
            teacher_batch_h, student_summary, batch
        )
        e_student_summary_student = self.discriminator_e_global(
            student_batch_h, student_summary, batch
        )
        prob_real_given_e_global = torch.sigmoid(e_student_summary_student)
        prob_fake_given_e_global = torch.sigmoid(e_teacher_summary_student)
        adverserial_global_e_loss2 = self.discriminator_loss(
            prob_real_given_e_global, torch.zeros_like(prob_real_given_e_global)
        ) + self.discriminator_loss(
            prob_fake_given_e_global, torch.ones_like(prob_fake_given_e_global)
        )
        student_loss = (
            student_loss
            + adversarial_local_e_loss
            + adverserial_global_e_loss1
            + adverserial_global_e_loss2
        )

        self.student_optimizer.zero_grad()
        student_loss.backward()
        self.student_optimizer.step()

        return student_loss.item()

    def train(self):
        best_valid_ap = 0
        for epoch in range(self.epochs):
            self.student_model.train()
            train_loss = 0
            for batch in self.train_loader:
                batch_loss = self._train_batch(batch, epoch)
                train_loss += batch_loss

            train_loss /= len(self.train_loader)

            if epoch % max(1, self.epochs // 10) == 0:
                valid_ap = self.evaluate(split="valid")
                print(
                    f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid AP: {valid_ap:.4f}",
                    flush=True,
                )

                # Save best model
                if valid_ap > best_valid_ap:
                    best_valid_ap = valid_ap
                    os.makedirs(f"{base_dir}/models", exist_ok=True)
                    torch.save(
                        self.student_model.state_dict(),
                        f"{base_dir}/models/gine_student_kd_{self.dataset_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pt",
                    )

    def evaluate(self, split="valid"):
        self.student_model.eval()
        loader = self.valid_loader if split == "valid" else self.test_loader
        y_true_list = []
        y_pred_list = []

        for batch in loader:
            batch = batch.to(self.device)
            if batch.x.shape[0] == 1:
                continue
            with torch.no_grad():
                y_pred = self.student_model(batch)
            y_true_list.append(batch.y.view(y_pred.shape).detach().cpu())
            y_pred_list.append(y_pred.detach().cpu())

        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_pred = torch.cat(y_pred_list, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        if self.dataset_name == "ogbg-molpcba":
            return self.evaluator.eval(input_dict)["ap"]
        else:
            return self.evaluator.eval(input_dict)["rocauc"]


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def run_multiple_experiments(
    teacher_knowledge_path,
    dataset_name="ogbg-molpcba",
    n_runs=5,
    include_vn_student=True,
    embedding_dim=400,
    student_lr=5e-3,
    student_weight_decay=1e-5,
    discriminator_lr=1e-2,
    discriminator_weight_decay=5e-4,
    batch_size=32,
    num_workers=4,
    discriminator_update_freq=5,
    epochs=100,
    output_file=f"{base_dir}/results/gine_student_gakd_molpcba.csv",
):
    results = []
    metric = "ap" if dataset_name == "ogbg-molpcba" else "rocauc"
    for run in range(n_runs):
        print(f"\nStarting Run {run + 1}/{n_runs}", flush=True)
        if include_vn_student:
            student_model = GINENetwork(
                hidden_dim=embedding_dim,
                out_dim=dataset_name.num_tasks,
                num_layers=3,
                dropout=0.5,
                virtual_node=True,
                train_vn_eps=False,
                vn_eps=0.0,
            )
        else:
            student_model = GINENetwork(
                hidden_dim=embedding_dim,
                out_dim=dataset_name.num_tasks,
                num_layers=3,
                dropout=0.5,
            )
        seed = 42 + run
        trainer = GAKD_trainer(
            student_model=student_model,
            teacher_knowledge_path=teacher_knowledge_path,
            dataset_name=dataset_name,
            embedding_dim=embedding_dim,
            student_lr=student_lr,
            student_weight_decay=student_weight_decay,
            discriminator_lr=discriminator_lr,
            discriminator_weight_decay=discriminator_weight_decay,
            batch_size=batch_size,
            num_workers=num_workers,
            discriminator_update_freq=discriminator_update_freq,
            epochs=epochs,
            seed=seed,
        )

        trainer.train()
        valid_ap = trainer.evaluate(split="valid")
        test_ap = trainer.evaluate(split="test")
        run_results = {
            "experiment_id": f"student_gine_gakd_{dataset_name}_{include_vn_student}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
            "dataset_name": dataset_name,
            "run": run + 1,
            "hidden_dim": embedding_dim,
            "virtual_node": include_vn_student,
            "n_params": numel(trainer.model, only_trainable=True),
            "lr": trainer.student_lr,
            "batch_size": trainer.batch_size,
            "epochs": trainer.epochs,
            "valid_metric": valid_ap,
            "test_metric": test_ap,
            "metric": metric,
        }
        results.append(run_results)

        # Save intermediate results after each run
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)

        print(f"Run {run + 1} Results:", flush=True)
        print(f"Validation {metric}: {valid_ap:.4f}", flush=True)
        print(f"Test {metric}: {test_ap:.4f}", flush=True)

    # Calculate and print summary statistics
    df = pd.DataFrame(results)
    summary = df[["valid_metric", "test_metric"]].agg(["mean", "std"])
    print(f"\nSummary Statistics:", flush=True)
    print(
        f"Validation {metric}: {summary['valid_metric']['mean']:.4f} ± {summary['valid_metric']['std']:.4f}",
        flush=True,
    )
    print(
        f"Test {metric}: {summary['test_metric']['mean']:.4f} ± {summary['test_metric']['std']:.4f}",
        flush=True,
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GINE experiments with or without virtual nodes under GAKD framework"
    )
    parser.add_argument(
        "--virtual_node",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to use virtual nodes in the student model",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["ogbg-molpcba", "ogbg-molhiv"],
        default="ogbg-molpcba",
        help="Name of the dataset to run experiments on",
    )
    # teacher knowledge path
    parser.add_argument(
        "--teacher_knowledge_path",
        type=str,
        default=f"{base_dir}/teacher_knowledge/teacher_knowledge_ogbg-molpcba.tar",
        help="Path to the teacher knowledge",
    )

    args = parser.parse_args()
    virtual_node = args.virtual_node.lower() == "true"

    os.makedirs(f"{base_dir}/results", exist_ok=True)
    experiment_type = "with" if virtual_node else "without"
    print(
        f"Running experiments {experiment_type} Virtual Nodes for {args.dataset_name}",
        flush=True,
    )
    results_df = run_multiple_experiments(
        args.teacher_knowledge_path,
        args.dataset_name,
        n_runs=5,
        include_vn_student=virtual_node,
        output_file=f"{base_dir}/results/gine_student_gakd_{args.dataset_name}_{experiment_type}.csv",
    )
    print(results_df.to_string(), flush=True)
    print("Experiments completed successfully!", flush=True)

