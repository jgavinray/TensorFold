def launch(R, N, K):
    return (((N + 63) // 64) * 64, R), (64, 1)
