def gradNorm(net, layer, alpha, dataloader, num_epochs, lr):
    """
    Args:
        net (nn.Module): a multitask network
        layer (nn.Module): subset of the full network layers where appling GradNorm on the weights
        alpha (float): hyperparameter of restoring force
        dataloader: (DataLoader): training dataloader
        epochs (int): number of epochs
    """
    # set optimizer
    optimizer1 = torch.optim.Adam(net.parameters(), lr=lr)
    # start traning
    iters = 0
    net.train()
    for epoch in range(num_epochs):
        # load data
        for data in dataloader:
            x, c, w, z = data
            # cuda
            if next(net.parameters()).is_cuda:
                data = [d.cuda() for d in data]
            # forward pass
            loss = net(*data)
            # initialization
            if iters == 0:
                # init weights
                weights = torch.ones_like(loss)
                weights = torch.nn.Parameter(weights / weights.sum())
                # set optimizer for weights
                optimizer2 = torch.optim.Adam([weights], lr=lr)
                # set L(0)
                l0 = loss.detach()
            # compute the weighted loss
            weighted_loss = weights @ loss
            # clear gradients of network
            optimizer1.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                dl = torch.autograd.grad(weights[i]*loss[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)
            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss 
            gradnorm_loss = torch.abs(gw - gw_avg * rt ** alpha).sum()
            # clear gradients of weights
            optimizer2.zero_grad()
            # backward pass for weigthted task loss
            gradnorm_loss.backward()
            # update model weights
            optimizer1.step()
            # update loss weights
            optimizer2.step()
            # renormalize weights
            weights = torch.nn.Parameter(weights / weights.sum())
            optimizer2 = torch.optim.Adam([weights], lr=lr)
            # update iters
            iters += 1
