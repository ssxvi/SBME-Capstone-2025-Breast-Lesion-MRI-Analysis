from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def build_metrics() -> Dict[str, List[float]]:
	# Hard-coded metrics from training logs
	epochs = list(range(1, 10))
	train_loss = [0.7946, 0.6811, 0.6336, 0.6149, 0.5779, 0.5274, 0.6825, 0.5444, 0.4998]
	val_pseudo_dice = [0.3472, 0.3848, 0.4318, 0.4092, 0.4003, 0.4048, 0.3936, 0.3750, 0.4460]
	val_f1 = [0.3704, 0.3876, 0.4370, 0.4144, 0.4042, 0.4269, 0.3966, 0.3826, 0.4482]
	val_auc = [0.9915, 0.9668, 0.9894, 0.9780, 0.9882, 0.9611, 0.9854, 0.9901, 0.9879]

	return {
		"epoch": epochs,
		"train_loss": train_loss,
		"val_pseudo_dice": val_pseudo_dice,
		"val_f1": val_f1,
		"val_auc": val_auc,
	}


def plot_metrics(metrics: Dict[str, List[float]], output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "training_metrics.png"

	epochs = metrics["epoch"]

	plt.style.use("seaborn-v0_8-darkgrid")
	fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

	ax.plot(epochs, metrics["train_loss"], marker="o", linewidth=2, label="Train Loss")
	ax.plot(epochs, metrics["val_pseudo_dice"], marker="o", linewidth=2, label="Val Pseudo Dice")
	ax.plot(epochs, metrics["val_f1"], marker="o", linewidth=2, label="Val F1")
	ax.plot(epochs, metrics["val_auc"], marker="o", linewidth=2, label="Val AUC")

	ax.set_xlabel("Epoch")
	ax.set_ylabel("Score / Loss")
	ax.set_title("Training Metrics over Epochs")
	ax.set_xticks(epochs)
	ax.set_ylim(0.0, 1.05)
	ax.legend(loc="best", frameon=True)
	ax.grid(True, linestyle="--", alpha=0.4)

	fig.tight_layout()
	fig.savefig(output_path)
	plt.close(fig)

	return output_path


def plot_individual_metrics(metrics: Dict[str, List[float]], output_dir: Path) -> Dict[str, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	epochs = metrics["epoch"]

	plt.style.use("seaborn-v0_8-darkgrid")

	def y_limits_for(key: str) -> tuple[float, float]:
		if key == "train_loss":
			values = metrics[key]
			vmin, vmax = min(values), max(values)
			margin = max(0.05, (vmax - vmin) * 0.2)
			return max(0.0, vmin - margin), vmax + margin
		return 0.0, 1.05

	names = {
		"train_loss": ("Train Loss", "training_loss.png"),
		"val_pseudo_dice": ("Validation Pseudo Dice", "val_pseudo_dice.png"),
		"val_f1": ("Validation F1", "val_f1.png"),
		"val_auc": ("Validation AUC", "val_auc.png"),
	}

	paths: Dict[str, Path] = {}
	for key, (pretty_name, filename) in names.items():
		fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
		ax.plot(epochs, metrics[key], marker="o", linewidth=2, label=pretty_name, color="#1f77b4")
		ax.set_xlabel("Epoch")
		ax.set_ylabel(pretty_name)
		ax.set_title(f"{pretty_name} over Epochs")
		ax.set_xticks(epochs)
		ymin, ymax = y_limits_for(key)
		ax.set_ylim(ymin, ymax)
		ax.grid(True, linestyle="--", alpha=0.4)
		ax.legend(loc="best", frameon=True)
		fig.tight_layout()

		out_path = output_dir / filename
		fig.savefig(out_path)
		plt.close(fig)
		paths[key] = out_path

	return paths


def print_best_epochs(metrics: Dict[str, List[float]]) -> None:
	def best_epoch_for(key: str, maximize: bool = True) -> int:
		values = metrics[key]
		return int(metrics["epoch"][values.index(max(values) if maximize else min(values))])

	best_loss_epoch = best_epoch_for("train_loss", maximize=False)
	best_dice_epoch = best_epoch_for("val_pseudo_dice", maximize=True)
	best_f1_epoch = best_epoch_for("val_f1", maximize=True)
	best_auc_epoch = best_epoch_for("val_auc", maximize=True)

	print("Best epochs:")
	print(f"  Train Loss (min): epoch {best_loss_epoch} -> {metrics['train_loss'][best_loss_epoch - 1]:.4f}")
	print(f"  Val Pseudo Dice (max): epoch {best_dice_epoch} -> {metrics['val_pseudo_dice'][best_dice_epoch - 1]:.4f}")
	print(f"  Val F1 (max): epoch {best_f1_epoch} -> {metrics['val_f1'][best_f1_epoch - 1]:.4f}")
	print(f"  Val AUC (max): epoch {best_auc_epoch} -> {metrics['val_auc'][best_auc_epoch - 1]:.4f}")


def main() -> None:
	metrics = build_metrics()
	out_dir = Path(__file__).parent
	out_path = plot_metrics(metrics, out_dir)
	individual_paths = plot_individual_metrics(metrics, out_dir)
	print_best_epochs(metrics)
	print(f"\nSaved plot to: {out_path.resolve()}")
	print("Saved individual plots:")
	for key, path in individual_paths.items():
		print(f"  {key}: {path.resolve()}")


if __name__ == "__main__":
	main()

