from .training_strategy import TrainingStrategy


class FullBatchStrategy(TrainingStrategy):
    """
    Batch GD clásico — un paso con TODOS los datos.

    W = W - η · (1/N) Σ_n ∂L_n/∂W


    Ventaja  : gradiente exacto sobre el dataset completo
    Desventaja: lento y costoso en memoria para datasets grandes,
                puede quedar atrapado en mínimos locales amplios
    """

    def step(self, model, X, y):
        y_pred = model.network.forward(X, training=True)
        loss = model.cost.function(y, y_pred)
        delta = model.cost.derivative(y, y_pred)

        model.network.backward(delta)
        model.optimizer.step(model.network.trainable_layers())

        return (
            float(loss),
            float(model.accuracy(y, y_pred)),
            model.compute_grad_norm(),
        )
