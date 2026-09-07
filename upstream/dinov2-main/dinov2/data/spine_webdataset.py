"""WebDataset input pipeline for SpinePretrain-v1 tar shards."""

from __future__ import annotations

import glob
import logging

logger = logging.getLogger("dinov2")


def make_spine_webdataset(
    shards: str,
    *,
    transform,
    shuffle_buffer: int = 5000,
    seed: int = 42,
):
    try:
        import webdataset as wds
    except ImportError as exc:
        raise RuntimeError(
            "WebDataset training requires: pip install webdataset"
        ) from exc

    if any(character in shards for character in "*?["):
        urls = sorted(glob.glob(shards))
    else:
        urls = wds.shardlists.expand_urls(shards)
    if not urls:
        raise FileNotFoundError(f"No WebDataset shards match: {shards}")
    logger.info("SpineWebDataset: %d shards, shuffle buffer=%d", len(urls), shuffle_buffer)

    def transform_sample(sample):
        image, _metadata = sample
        return transform(image), ()

    return (
        wds.WebDataset(
            urls,
            resampled=True,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.warn_and_continue,
            seed=seed,
        )
        .shuffle(shuffle_buffer, initial=min(1000, shuffle_buffer))
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("png;jpg;jpeg;tif;tiff", "json")
        .map(transform_sample, handler=wds.warn_and_continue)
    )
