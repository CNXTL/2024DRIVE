 
import torch
import torch.nn as nn
import time
class PGD_input(nn.Module):
    def __init__(self, model, loss_function, bs, eps=0.08, alpha=0.0015, steps=5, random_start=True, device='cuda:0', pred_len=48):
        super(PGD_input, self).__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.pred_len = pred_len
        self.device = device
        self.calculate_loss = loss_function
        self.bs = bs
        self.model.eval()

    def forward(self, batch):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch

        image_array.requires_grad = True
        inputs = image_array.clone().to(self.device).detach().requires_grad_(True)

        gt_angle = angle.clone().to(self.device)
        gt_distance = distance.clone().to(self.device)
        gt_vego = vego.clone().to(self.device)
        loss = self.calculate_loss  # Change to MSELoss for regression tasks

        noise_img = inputs.clone().to(self.device).detach().requires_grad_(True)  

        if self.random_start:
            init_noise = torch.randn_like(inputs) * 0.03 
            noise_img = noise_img + init_noise
            noise_img = torch.clamp(noise_img, min=0, max=1)  # restrain noise strength
        timex =time.time()
        for j in range(self.steps):
            
            noise_img.grad = None
            # if self.multitask != "multitask":
            #     logits, attns, concepts = self.model(noise_img, gt_angle, gt_distance, gt_vego)
            # else:
            timea = time.time()
            logits, attns, concepts = self.model(noise_img, gt_angle, gt_distance, gt_vego)
            timeb =time.time()

            cost = loss(logits, gt_angle, gt_distance)
            timec=time.time()
            grad = torch.autograd.grad(cost, noise_img, retain_graph=False, create_graph=False)[0]
            timed=time.time()
            noise_img = noise_img.detach() + self.alpha * grad.sign()
            delta = torch.clamp(inputs - noise_img, min=-self.eps, max=self.eps)
            noise_img = (delta + inputs).detach().requires_grad_(True)
        timey =time.time()
        return noise_img

    def perturb(self, batch):
        self.model.eval()
        with torch.enable_grad():  # 启用梯度计算
            noise_img = self.forward(batch)
        return noise_img

        
    
class PGD_layer(nn.Module):
    def __init__(self, model,loss_function,bs, eps=0.08, alpha=0.003, steps=30, random_start=True, device='cuda:0', pred_len=48):
        super(PGD_layer, self).__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.pred_len = pred_len
        self.device = device
        self.calculate_loss = loss_function
        self.bs=bs
    

    def forward(self, batch):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        inputs = image_array.clone().detach().to(self.device)
        gt_angle = angle.clone().detach().to(self.device)
        gt_distance = distance.clone().detach().to(self.device)
        gt_vego =vego.clone().detach().to(self.device)
        bs =self.bs
        loss = self.calculate_loss  # Change to MSELoss for regression tasks
        img = inputs.clone().detach()

        
        if self.random_start:
            init_noise = torch.randn(bs,240,100)
            stddev = 0.01 

            layer_noise = (init_noise * stddev).to(self.device)
            # layer_noise.to(self.device)


            # noise = perturb(adv_img)
            #nosiy = torch.randn_like(adv_inputs ) * 0.1#self.eps
            # adv_img = adv_img + noise * stddev
            # (torch.rand(adv_inputs.size(), device=self.device) * 2 * self.eps - self.eps)
            # torch.normal(mean=0.0, std=self.eps, size=adv_inputs.size()).to(self.device)
            # adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach()
            img = img.detach()

        for _ in range(self.steps):
            img.requires_grad = True
            logits, attns,concepts= self.model(img, gt_angle, gt_distance, gt_vego)
    
            cost = loss(logits, gt_angle,gt_distance)

            grad = torch.autograd.grad(cost, layer_noise, retain_graph=False, create_graph=False)
            
            layer_noise = layer_noise.detach() + self.alpha * grad.sign()
            delta = torch.clamp(layer_noise - init_noise * stddev, min=-self.eps, max=self.eps)
            #adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()
            #adv_img = (inputs + delta).detach()

        return delta 
        




