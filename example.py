import torch
x=torch.Tensor([[[1,2],[3,4]]])
'''(N, S, E)  --> (N, 1, S, S)'''
'''S 帧数 N batch E embeding'''
f = x.shape[1]

I = torch.ones(f)

xr = torch.einsum('bfe,h->bhfe', (x, I))  # [x, x, x, x ....]  =>  xr[:,0,:,:] == x
xc = torch.einsum('bfe,h->bfhe', (x, I))  # [x x x x ....]     =>  xc[:,:,0,:] == x
diff = torch.sub(xr, xc).contiguous()
out = torch.einsum('bfge,bfge->bfg', (diff, diff))
out = out.unsqueeze(1)