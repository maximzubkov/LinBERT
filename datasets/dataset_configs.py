datasets = {
    "agnews": {
        "columns": {"type": "agnews", "feature": "content", "target": "class"},
        "names": ["class", "title", "content"],
        "num_labels": 4,
        "max_length": 128,
    },
    "dbpedia": {
        "columns": {"type": "dbpedia", "feature": "content", "target": "class"},
        "names": ["class", "title", "content"],
        "num_labels": 14,
        "max_length": 128,
    },
    "yelp_full": {
        "columns": {"type": "yelp_full", "feature": "text", "target": "class"},
        "names": ["class", "text"],
        "num_labels": 5,
        "max_length": 128,
    },
    "yelp_polarity": {
        "columns": {"type": "yelp_full", "feature": "text", "target": "class"},
        "names": ["class", "text"],
        "num_labels": 2,
        "max_length": 128,
    },
}
