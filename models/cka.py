import torch


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return H @ K @H(()‘•˜É‰˜¡`°Í¥µ„õ9½¹”¤è(€€€`€ôÑ½É ¹•¥¹ÍÕ´ ‰¤±¤´ù‰Œœ°`°`¤(€€€-`€ôÑ½É ¹‘¥…œ¡`¤€´€È€¨`€¬Ñ½É ¹‘¥…œ¡`¤¹P(€€€¥˜Í¥µ„¥Ì9½¹”è(€€€€€€€µ‘¥ÍĞ€ôÑ½É ¹µ•‘¥…¸¡-alKX != 0])
        sigma = torch.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def cKA(X, Y, sigma=None):
    """
    Computes Centered Kernel Alignment (CKA) between two matrices X and Y.
    """
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    # Handle cases where variance is zero
    if var1 * var2 == 0:
        return torch.tensor(0.0, device=X.device)

    return hsic / (var1 * var2)
