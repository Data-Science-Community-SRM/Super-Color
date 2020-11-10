from time import time as Tim


def train(model, train_loader, criterion, optimizer, num_epochs,device):
    startTime = Tim()
    for e in range(num_epochs):
        i = 0
        for images, out in train_loader:
            images = images.to(device)
            out = out.to(device)

            optimizer.zero_grad()
            
            output = model(images)

            loss = criterion(output,out)
            
            loss.backward()

            if i % 200 == 0:
                print("e : %2d ,iter : %4d , loss: %3.10f , t : %3.5f min"%(e,i,loss.item(),(Tim() - startTime)/60))
                                
            i += 1
            optimizer.step()