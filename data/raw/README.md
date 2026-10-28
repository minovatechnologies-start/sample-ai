# Raw dataset input

Place approved `.jsonl`, `.json`, or `.csv` source datasets in this folder. Keep the source license, source URL, and repository/revision provenance with each record. See the dataset schema and license guidance in the project [README](../../README.md).

The training pipeline reads all supported files under this folder. Synthetic samples are kept separately in `data/examples` so they are not accidentally included in training.
