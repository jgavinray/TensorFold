def launch(kernel, n):
    return {"embed": ((1024, n), (64, 1)), "head_norm": ((64 * n,), (64,)), "queries": ((256, 8, n), (256, 1, 1)),
            "keys": ((256, 2, n), (256, 1, 1)), "attention_gate": ((2048, n), (256, 1))}[kernel]
