from torch import nn, optim
import matplotlib.pyplot as plt


def grad(f, *x):
    for xi in x:
        xi.requires_grad_() # això indica a PyTorch que ha de calcular el gradient d'aquest tensor
        xi.grad = None # resetejem el gradient
    
    y = f(*x) # calculem el valor de la funció
    assert y.shape == (len(y),) # assegurem-nos que cada valor dona un escalar com a resultat
    escalar = y.sum() # si sumem tots els resultats, calcularem el gradient per cada entrada individual
    escalar.backward() # li diem a PyTorch que calculi el gradient a partir d'aquest escalar
    
    # El nostre resultat està a x.grad
    return tuple(xi.grad for xi in x)


def minimize(f, *parameters, n_epochs=1000, plot=True, optimizer=None, **optim_kwargs):
    assert optimizer is not None, "Has d'indicar quin optimizer utilitzar"
    
    parameters = [ nn.Parameter(p) for p in parameters ]
    optimizer = optimizer(parameters, **optim_kwargs)
    losses = []
    
    # Entrenem els paràmetres n_epochs vegades
    for epoch in range(1, n_epochs + 1):
        
        # Calculem el gradient:
        optimizer.zero_grad() # això reseteja el gradient de tots els paràmetres
        current_loss = f(*parameters) # params és una llista, hem de fer unpack
        assert len(current_loss.shape) < 2, 'f ha de retornar outputs escalars'
        current_loss = current_loss.sum() # recordeu transformar a un únic escalar amb sum
        current_loss.backward() # creem el gradient
        
        # Fem el pas del gradient mitjançant la fòrmula de l'optimizer
        optimizer.step() 
        
        losses.append(current_loss.item()) # guardem les losses per pintar-les a continuació
      
    if plot:
        plt.figure()
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')

    return [ p.data for p in parameters ] # fem .data per obtenir un tensor normal, no un nn.Parameter


def train_module(module, X, Y, loss_f, n_epochs=1000, optimizer=optim.Adam, plot=True, **optim_kwargs):
    assert optimizer is not None, "Has d'indicar quin optimizer utilitzar"
    optimizer = optimizer(module.parameters(), **optim_kwargs)
    
    losses = []
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = loss_f(Y, module.predict(X))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    if plot:
        plt.figure()
        plt.plot(range(1, n_epochs + 1), losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')