RANDOM_SEED = 42

TARGET_COLUMN = "resolution"
ROW_ID_COLUMN = "row_id"
ITEM_ID_COLUMN = "ItemID"
SELLER_ID_COLUMN = "SellerID"

DEFAULT_TEXT_FIELDS = (
    "brand_name",
    "name_rus",
    "CommercialTypeName4",
    "description",
)

DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

DEFAULT_TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_IMAGE_MODEL_NAME = "efficientnet_b0"
DEFAULT_IMAGE_EMBED_DIM = 1280

EXCLUDED_NUMERIC_COLUMNS = {
    TARGET_COLUMN,
    "id",
    ITEM_ID_COLUMN,
    SELLER_ID_COLUMN,
}

LOG1P_COLUMN_HINTS = (
    "count",
    "gmv",
    "price",
    "alive",
    "value",
    "accepted",
    "returned",
    "sales",
    "returns",
    "available",
    "variety",
)
