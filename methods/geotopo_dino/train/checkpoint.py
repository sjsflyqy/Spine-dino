"""Reject accidental decoder architecture changes before restoring any state."""

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


class GeoTopoCheckpointer(FSDPCheckpointer):
    def save(self, name, **kwargs):
        signature = self.model.pixel_reconstruction_signature
        if signature["enabled"]:
            kwargs["pixel_reconstruction_signature"] = signature
        super().save(name, **kwargs)

    def _load_model(self, checkpoint):
        validate_pixel_checkpoint(checkpoint, self.model.pixel_reconstruction_signature)
        return super()._load_model(checkpoint)
