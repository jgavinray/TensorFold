def launch(W):
    # 128 work-items per (head, row): one work-item per channel of the head, 48 heads.
    return (128, 48, W), (128, 1, 1)
