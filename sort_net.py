# sort_5_numbers.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
from datetime import datetime

# ============= ПАРАМЕТРЫ ДЛЯ 5 ЭЛЕМЕНТОВ =============
ARRAY_LENGTH = 5               # Длина массива
MAX_VALUE = 100                # Числа от 1 до 100
NUM_SAMPLES = 100000           # Количество примеров
EPOCHS = 50                    # Больше эпох из-за сложности
BATCH_SIZE = 256               # Увеличим для стабильности
LEARNING_RATE = 0.001
HIDDEN_LAYERS = [1024, 1024, 512, 256]  # Глубже для сложности
DROPOUT = 0.2
# =====================================================

INPUT_SIZE = ARRAY_LENGTH * MAX_VALUE   # 500
OUTPUT_SIZE = ARRAY_LENGTH * MAX_VALUE  # 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Используется устройство: {device}")

class SortNet5(nn.Module):
    """Нейросеть для сортировки 5 чисел"""
    def __init__(self):
        super().__init__()

        layers = []
        in_features = INPUT_SIZE

        for hidden_size in HIDDEN_LAYERS:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(DROPOUT)
            ])
            in_features = hidden_size

        layers.append(nn.Linear(in_features, OUTPUT_SIZE))
        self.network = nn.Sequential(*layers)

        # Инициализация
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

def prepare_data_5():
    """Подготовка данных для 5-элементных массивов"""
    X = np.zeros((NUM_SAMPLES, INPUT_SIZE), dtype=np.float32)
    y = np.zeros((NUM_SAMPLES, OUTPUT_SIZE), dtype=np.float32)

    print(f"  Генерация {NUM_SAMPLES} примеров...", end='', flush=True)
    start_time = time.time()

    for i in range(NUM_SAMPLES):
        # Генерируем 5 случайных чисел
        arr = np.random.randint(1, MAX_VALUE + 1, size=ARRAY_LENGTH)
        sorted_arr = np.sort(arr)

        # One-hot encoding для входа
        for pos, val in enumerate(arr):
            X[i, pos * MAX_VALUE + (val - 1)] = 1.0

        # One-hot encoding для выхода (отсортированный)
        for pos, val in enumerate(sorted_arr):
            y[i, pos * MAX_VALUE + (val - 1)] = 1.0

    gen_time = time.time() - start_time
    print(f" готово за {gen_time:.2f} сек")

    # Перемешиваем
    indices = np.random.permutation(NUM_SAMPLES)
    X = X[indices]
    y = y[indices]

    return X, y

def calculate_accuracy(outputs, targets):
    """Вычисляет точность сортировки"""
    # outputs: [batch, 500], targets: [batch, 500]
    batch_size = outputs.size(0)

    # Получаем предсказанные числа для каждой позиции
    pred_nums = torch.zeros(batch_size, ARRAY_LENGTH, dtype=torch.long, device=device)
    true_nums = torch.zeros(batch_size, ARRAY_LENGTH, dtype=torch.long, device=device)

    for pos in range(ARRAY_LENGTH):
        start = pos * MAX_VALUE
        pred_nums[:, pos] = outputs[:, start:start+MAX_VALUE].argmax(dim=1)
        true_nums[:, pos] = targets[:, start:start+MAX_VALUE].argmax(dim=1)

    # Сравниваем поэлементно
    correct_elements = (pred_nums == true_nums).float()

    # Полностью правильные массивы (все 5 позиций)
    correct_arrays = (correct_elements.sum(dim=1) == ARRAY_LENGTH).float()

    return {
        'array_accuracy': correct_arrays.mean().item(),
        'element_accuracy': correct_elements.mean().item(),
        'per_position': correct_elements.mean(dim=0).cpu().numpy()
    }

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_arrays = 0
    total_elements = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_arrays += X_batch.size(0)

    return total_loss / total_arrays

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_arrays = 0

    # Для сбора метрик точности
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_arrays += X_batch.size(0)

            all_outputs.append(outputs.cpu())
            all_targets.append(y_batch.cpu())

    # Конкатенируем и считаем точность
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_accuracy(all_outputs, all_targets)

    return total_loss / total_arrays, metrics

def main():
    print("="*80)
    print("🧠 ТРЕНИРОВКА ДЛЯ СОРТИРОВКИ 5 ЧИСЕЛ")
    print("="*80)

    print(f"\n📊 Статистика задачи:")
    print(f"  Всего возможных массивов (с повторениями): {100**ARRAY_LENGTH:,}")
    print(f"  Обучающих примеров: {NUM_SAMPLES:,}")
    print(f"  Покрытие: {NUM_SAMPLES / (100**ARRAY_LENGTH) * 100:.8f}%")
    print(f"  (всего 0.001% от пространства)")

    print(f"\n📋 Параметры модели:")
    print(f"  Размер входа: {INPUT_SIZE}")
    print(f"  Скрытые слои: {HIDDEN_LAYERS}")
    print(f"  Параметров: ~{sum(l*next for l,next in zip([INPUT_SIZE]+HIDDEN_LAYERS, HIDDEN_LAYERS+[OUTPUT_SIZE])):,}")

    # Подготовка данных
    print("\n📦 Подготовка данных...")
    X, y = prepare_data_5()

    # Конвертация в тензоры
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Разбиение на train/val/test
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"\n  Train: {train_size:,} примеров")
    print(f"  Val:   {val_size:,} примеров")
    print(f"  Test:  {test_size:,} примеров")

    # Модель
    model = SortNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n🏋️ Начало обучения на {device}...")
    print("-"*90)
    print(f"{'Эпоха':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Array Acc':>8} | {'Elem Acc':>8} | {'P1':>5} | {'P2':>5} | {'P3':>5} | {'P4':>5} | {'P5':>5}")
    print("-"*90)

    best_val_acc = 0

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion)

        scheduler.step()

        # Вывод каждые 5 эпох
        if (epoch + 1) % 5 == 0 or epoch == 0:
            pos_acc = val_metrics['per_position']
            print(f"{epoch+1:6d} | {train_loss:10.6f} | {val_loss:8.6f} | "
                  f"{val_metrics['array_accuracy']:8.4f} | {val_metrics['element_accuracy']:8.4f} | "
                  f"{pos_acc[0]:5.3f} | {pos_acc[1]:5.3f} | {pos_acc[2]:5.3f} | {pos_acc[3]:5.3f} | {pos_acc[4]:5.3f}")

        # Сохраняем лучшую модель
        if val_metrics['array_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['array_accuracy']
            torch.save(model.state_dict(), f"models/best_model_5_{datetime.now().strftime('%Y%m%d')}.pt")

    print("-"*90)
    print(f"\n✅ Лучшая точность на валидации: {best_val_acc:.4f}")

    # Финальное тестирование
    print("\n🧪 Тестирование на отложенной выборке...")
    test_loss, test_metrics = validate(model, test_loader, criterion)
    print(f"\n📊 РЕЗУЛЬТАТЫ НА ТЕСТЕ:")
    print(f"  Точность полных массивов: {test_metrics['array_accuracy']*100:.2f}%")
    print(f"  Точность элементов: {test_metrics['element_accuracy']*100:.2f}%")
    print(f"\n  По позициям:")
    for i, acc in enumerate(test_metrics['per_position']):
        print(f"    Позиция {i+1}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
