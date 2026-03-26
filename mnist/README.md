# MNIST Training Scripts

Conjunto de scripts para entrenar redes neuronales en el dataset MNIST con diferentes estrategias de entrenamiento.

## Requisitos

```bash
pip install -r requirements.txt
```

## Scripts

### main.py
Entrenamiento básico secuencial con `WeightAvgStrategy`.

```bash
python -m mnist.main
```

### parallel.py
Entrenamiento paralelo usando múltiples procesos con `ParallelWeightAvgStrategy`.

```bash
python -m mnist.parallel
```

### repetitions.py
Ejecuta 20 repeticiones comparando:
- `WeightAvgStrategy` (batch averaging)
- `FinalAvgStrategy` (entrenamiento completo por worker, luego promedio)

```bash
python -m mnist.repetitions
```

### repetitions_parallel.py
Similar a `repetitions.py` pero con `ParallelWeightAvgStrategy` para paralelización real.

```bash
python -m mnist.repetitions_parallel
```

### distributed.py
Entrenamiento distribuido cliente-servidor con **Gradient Averaging**.

| Modo | Ejecución |
|------|-----------|
| Local | `python -m mnist.distributed --local --workers 2 --epochs 200` |
| Servidor | `python -m mnist.distributed --server --host 0.0.0.0 --port 9999 --workers 2` |
| Cliente | `python -m mnist.distributed --client --host IP_SERVIDOR --port 9999` |

**Parámetros:**
- `--lr` - Learning rate (default: 0.05)
- `--epochs` - Número de épocas (default: 200)
- `--workers` - Número de workers (default: 1)
- `--min_workers` - Mínimo workers para iniciar (servidor, default: 1)

## Parámetros comunes de experimentación

| Variable | Default | Descripción |
|----------|---------|-------------|
| `REPETITIONS` | 20 | Repeticiones por estrategia |
| `EPOCHS` | 100 | Épocas de entrenamiento |
| `LR` | 0.1 | Learning rate |
| `BATCH_SIZE` | 6000 | Tamaño de batch |
| `N_WORKERS` | 10 | Número de workers |
