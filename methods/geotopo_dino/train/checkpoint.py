"""Reject decoder/objective changes before restoring training state."""

from dinov2.fsdp import FSDPCheckpointer


def validate_pixel_checkpoint(checkpoint, expected):
    saved = checkpoint.get("pixel_reconstruction_signature")
    if saved is None:
        has_decoder = any("student_aux.pixel_decoder." in key for key in checkpoint.get("model", {}))
        saved = {"enabled": has_decoder}
    if saved != expected:
        raise ValueError(
            "Pixel reconstruction checkpoint/config mismatch. Resume with the same enabled flag, "
            "decoder architecture, and norm_pix_loss. To change these, start a new output directory "
            "with --no-resume and initialize the backbone via student.pretrained_weights; "
            "do not use MODEL.WEIGHTS for a decoder-incompatible training checkpoint. "
            f"Saved: {saved}; requested: {expected}"
        )


def validate_wavelet_checkpoint(checkpoint, expected):
    # Legacy F025 checkpoints predate wavelet supervision and remain resumable
    # with wavelet disabled. A new objective must start a separate experiment.
    saved = checkpoint.get("wavelet_reconstruction_signature", {"enabled": False})
    if saved != expected:
        raise ValueError(
            "Wavelet reconstruction checkpoint/config mismatch. Resume with the same enabled flag, "
            "loss_weight, and warmup_iterations. To change the objective, start a new output directory "
            "with --no-resume and initialize via student.pretrained_weights, not MODEL.WEIGHTS. "
            f"Saved: {saved}; requested: {expected}"
        )


def validate_gradient_checkpoint(checkpoint, expected):
    saved = checkpoint.get("gradient_reconstruction_signature", {"enabled": False})
    if saved != expected:
        raise ValueError(
            "Gradient reconstruction checkpoint/config mismatch. Resume with the same enabled flag, "
            "Sobel settings, loss_weight, and warmup_iterations. To change the objective, start a new "
            "output directory with --no-resume and initialize via student.pretrained_weights, "
            "not MODEL.WEIGHTS. "
            f"Saved: {saved}; requested: {expected}"
        )


class GeoTopoCheckpointer(FSDPCheckpointer):
    def save(self, name, **kwargs):
        signature = self.model.pixel_reconstruction_signature
        if signature["enabled"]:
            kwargs["pixel_reconstruction_signature"] = signature
        wavelet_signature = self.model.wavelet_reconstruction_signature
        if wavelet_signature["enabled"]:
            kwargs["wavelet_reconstruction_signature"] = wavelet_signature
        gradient_signature = self.model.gradient_reconstruction_signature
        if gradient_signature["enabled"]:
            kwargs["gradient_reconstruction_signature"] = gradient_signature
        super().save(name, **kwargs)

    def _load_model(self, checkpoint):
        validate_pixel_checkpoint(checkpoint, self.model.pixel_reconstruction_signature)
        validate_wavelet_checkpoint(checkpoint, self.model.wavelet_reconstruction_signature)
        validate_gradient_checkpoint(checkpoint, self.model.gradient_reconstruction_signature)
        return super()._load_model(checkpoint)
