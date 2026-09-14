"""
train_utils.py

Deljena funkcija za trening modela sa early stopping-om.
Koristi se i za baseline i za AutoML finalni model, da bi poređenje bilo fer
(isti postupak treniranja za oba).
"""

import torch


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                                device, max_epochs=60, patience=10):
    """
    Trening petlja sa early stopping-om.

    Prati validacioni loss kroz epohe. Ako se val_loss NE POBOLJŠA kroz
    'patience' uzastopnih epoha, trening se PREKIDA - dalje treniranje
    bi samo overfitovalo (videli smo taj obrazac kod oba modela do sad).

    Čuva NAJBOLJE težine modela (na osnovu najnižeg val_loss), ne poslednje
    epohe - ovo rešava problem koji smo primetili (poslednja epoha nije
    uvek najbolja).

    Vraća:
        history - dict sa listama metrika kroz sve odrađene epohe (za grafik)
        best_model_state - state_dict NAJBOLJE verzije modela
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(max_epochs):
                              
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_specs, batch_labels in train_loader:
            batch_specs = batch_specs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_specs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_specs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

                                  
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_specs, batch_labels in val_loader:
                batch_specs = batch_specs.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_specs)
                loss = criterion(outputs, batch_labels)

                val_running_loss += loss.item() * batch_specs.size(0)
                _, predicted = torch.max(outputs, dim=1)
                val_correct += (predicted == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

                                       
        if epoch_val_loss < best_val_loss:
                                                                             
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
                                                                              
                                                                 
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " <- NOVI NAJBOLJI"
        else:
            epochs_without_improvement += 1
            marker = f" ({epochs_without_improvement}/{patience} bez poboljšanja)"

        print(f"Epoha {epoch+1}/{max_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}{marker}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping - nema poboljšanja kroz {patience} epoha. "
                  f"Zaustavljamo na epohi {epoch+1}.")
            break

    return history, best_model_state