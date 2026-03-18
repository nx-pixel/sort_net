# test_5_numbers.py
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from datetime import datetime

# ============= ПАРАМЕТРЫ (ДОЛЖНЫ СОВПАДАТЬ С ТРЕНИРОВОЧНЫМИ) =============
ARRAY_LENGTH = 5              # Длина массива
MAX_VALUE = 100                # Числа от 1 до 100
HIDDEN_LAYERS = [1024, 1024, 512, 256]  # Размеры скрытых слоёв
DROPOUT = 0.2                  # Dropout
# =========================================================================

INPUT_SIZE = ARRAY_LENGTH * MAX_VALUE   # 500
OUTPUT_SIZE = ARRAY_LENGTH * MAX_VALUE  # 500

# Определяем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Используется устройство: {device}")

class SortNet5(nn.Module):
    """Та же архитектура, что и в обучающей программе для 5 чисел"""
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

    def forward(self, x):
        return self.network(x)

def find_latest_model(models_dir="models"):
    """Находит самую свежую модель для 5 чисел"""
    # Ищем модели для 5 чисел
    model_files = glob.glob(os.path.join(models_dir, "best_model_5_*.pt"))
    model_files += glob.glob(os.path.join(models_dir, "sortnet_5_*.pt"))
    model_files += glob.glob(os.path.join(models_dir, "*.pt"))

    if not model_files:
        return None

    # Сортируем по времени создания (самые новые - последние)
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

def load_model(model_path=None):
    """Загружает модель (последнюю или указанную)"""
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("❌ Модель не найдена! Сначала обучите модель.")
            return None, None

    print(f"📂 Загрузка модели: {model_path}")

    try:
        # Пробуем загрузить как полный checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        model = SortNet5().to(device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Модель загружена из checkpoint!")
            if 'params' in checkpoint:
                print(f"   Параметры: {checkpoint['params']}")
        else:
            # Если это просто state_dict
            model.load_state_dict(checkpoint)
            print("✅ Модель загружена из state_dict!")

        model.eval()
        return model, checkpoint

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None, None

def encode_array(arr):
    """Кодирует массив в one-hot формат"""
    x = np.zeros((1, INPUT_SIZE), dtype=np.float32)
    for i, val in enumerate(arr):
        if 1 <= val <= MAX_VALUE:
            x[0, i * MAX_VALUE + (val - 1)] = 1.0
        else:
            raise ValueError(f"Число {val} вне диапазона 1-{MAX_VALUE}")
    return x

def decode_prediction(pred_tensor):
    """Декодирует выход нейросети в массив чисел"""
    pred_np = pred_tensor.cpu().detach().numpy()
    result = []
    for i in range(ARRAY_LENGTH):
        start = i * MAX_VALUE
        segment = pred_np[0, start:start + MAX_VALUE]
        val = np.argmax(segment) + 1
        result.append(int(val))
    return result

def predict(model, arr):
    """Предсказывает сортировку для одного массива"""
    x = encode_array(arr)
    x_tensor = torch.FloatTensor(x).to(device)

    with torch.no_grad():
        output = model(x_tensor)
        predicted = decode_prediction(output)

    return predicted

def get_confidence(model, arr):
    """Возвращает уверенность модели для каждого предсказания"""
    x = encode_array(arr)
    x_tensor = torch.FloatTensor(x).to(device)

    with torch.no_grad():
        output = model(x_tensor)
        confidences = []
        for i in range(ARRAY_LENGTH):
            start = i * MAX_VALUE
            segment = output[0, start:start + MAX_VALUE].cpu().numpy()
            # Нормализуем до вероятностей (softmax)
            exp_segment = np.exp(segment - np.max(segment))
            probs = exp_segment / exp_segment.sum()
            conf = np.max(probs)
            confidences.append(float(conf))
    return confidences

def manual_testing(model):
    """Режим ручного ввода массивов"""
    print("\n" + "="*70)
    print("🎮 РЕЖИМ РУЧНОГО ТЕСТИРОВАНИЯ (5 ЧИСЕЛ)")
    print("="*70)
    print(f"Введите {ARRAY_LENGTH} чисел от 1 до {MAX_VALUE} через пробел")
    print("Пример: 23 67 12 89 45")
    print("Команды:")
    print("  'q'  - выход в меню")
    print("  'h'  - показать историю")
    print("  'c'  - очистить историю")

    history = []

    while True:
        print("\n" + "-"*70)
        user_input = input("Числа: ").strip()

        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 'h':
            if history:
                print("\n📜 История последних тестов:")
                print(f"{'№':>3} | {'Входной массив':<25} | {'Предсказание':<20} | {'Результат'}")
                print("-"*70)
                for i, (arr, pred, correct, status) in enumerate(history[-10:], 1):
                    arr_str = str(arr)
                    pred_str = str(pred)
                    print(f"{i:3d} | {arr_str:<25} | {pred_str:<20} | {status}")
            else:
                print("  История пуста")
            continue
        elif user_input.lower() == 'c':
            history = []
            print("  История очищена")
            continue

        try:
            numbers = list(map(int, user_input.split()))

            if len(numbers) != ARRAY_LENGTH:
                print(f"❌ Нужно ввести ровно {ARRAY_LENGTH} чисел")
                continue

            if not all(1 <= n <= MAX_VALUE for n in numbers):
                print(f"❌ Числа должны быть от 1 до {MAX_VALUE}")
                continue

            # Предсказание
            predicted = predict(model, numbers)
            correct = sorted(numbers)

            # Результат
            is_correct = (predicted == correct)
            status = "✅ ВЕРНО" if is_correct else "❌ НЕВЕРНО"

            # Уверенность
            confidences = get_confidence(model, numbers)

            print(f"\n  Входные:     {numbers}")
            print(f"  Предсказано: {predicted}")
            print(f"  Правильно:   {correct}")
            print(f"  Результат:   {status}")
            print(f"  Уверенность: {[f'{c:.3f}' for c in confidences]}")

            # Показываем, какие позиции ошибочны
            if not is_correct:
                errors = []
                for pos in range(ARRAY_LENGTH):
                    if predicted[pos] != correct[pos]:
                        errors.append(f"поз.{pos+1}: {predicted[pos]}≠{correct[pos]}")
                print(f"  Ошибки:      {', '.join(errors)}")

            # Сохраняем в историю
            history.append((numbers, predicted, correct, status))

        except ValueError:
            print("❌ Ошибка: введите числа через пробел")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

def batch_testing(model):
    """Массовое тестирование на случайных массивах"""
    print("\n" + "="*70)
    print("📊 МАССОВОЕ ТЕСТИРОВАНИЕ")
    print("="*70)

    try:
        num_tests = int(input("Сколько примеров сгенерировать? (рекомендуется 1000-10000): ").strip())
        if num_tests <= 0:
            print("❌ Введите положительное число")
            return
    except ValueError:
        print("❌ Введите число")
        return

    print(f"\nТестирование на {num_tests} случайных массивах...")
    print("-"*70)

    correct_count = 0
    position_correct = [0] * ARRAY_LENGTH
    total_elements = num_tests * ARRAY_LENGTH

    # Для анализа ошибок
    error_log = []

    for test_num in range(num_tests):
        # Генерируем случайный массив
        arr = np.random.randint(1, MAX_VALUE + 1, size=ARRAY_LENGTH)
        correct = sorted(arr)

        # Предсказание
        predicted = predict(model, arr)

        # Проверка массива целиком
        if predicted == correct:
            correct_count += 1
        else:
            # Логируем первые 10 ошибок для анализа
            if len(error_log) < 10:
                error_log.append((arr.tolist(), predicted, correct))

        # Проверка каждой позиции
        for pos in range(ARRAY_LENGTH):
            if predicted[pos] == correct[pos]:
                position_correct[pos] += 1

        # Прогресс
        if (test_num + 1) % (num_tests // 10) == 0:
            print(f"  Прогресс: {int((test_num+1)/num_tests*100)}%")

    # Результаты
    print("\n" + "="*70)
    print("📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*70)

    array_accuracy = correct_count / num_tests * 100
    element_accuracy = sum(position_correct) / total_elements * 100

    print(f"\n✅ Полностью верных массивов: {correct_count}/{num_tests} ({array_accuracy:.2f}%)")
    print(f"✅ Верных элементов: {sum(position_correct)}/{total_elements} ({element_accuracy:.2f}%)")

    print(f"\n📊 Точность по позициям:")
    for pos in range(ARRAY_LENGTH):
        pos_acc = position_correct[pos] / num_tests * 100
        print(f"  Позиция {pos+1}: {position_correct[pos]}/{num_tests} ({pos_acc:.2f}%)")

    # Анализ ошибок
    if error_log:
        print(f"\n🔍 Примеры ошибок (первые 10):")
        for i, (arr, pred, correct) in enumerate(error_log, 1):
            print(f"\n  {i}. Вход:     {arr}")
            print(f"     Предсказано: {pred}")
            print(f"     Правильно:   {correct}")
            # Подсвечиваем ошибки
            diff = []
            for pos in range(ARRAY_LENGTH):
                if pred[pos] != correct[pos]:
                    diff.append(f"поз.{pos+1}: {pred[pos]}→{correct[pos]}")
            if diff:
                print(f"     Ошибки: {', '.join(diff)}")

    return array_accuracy

def interactive_menu(model):
    """Интерактивное меню"""
    while True:
        print("\n" + "="*70)
        print("🏴‍☠️ ГЛАВНОЕ МЕНЮ ТЕСТИРОВАНИЯ (5 ЧИСЕЛ)")
        print("="*70)
        print("1. 🎮 Ручной ввод массивов")
        print("2. 📊 Массовое тестирование")
        print("3. 🔍 Показать информацию о модели")
        print("4. 📋 Список доступных моделей")
        print("5. 🔄 Загрузить другую модель")
        print("6. 🚪 Выход")
        print("-"*70)

        choice = input("Выберите режим (1-6): ").strip()

        if choice == '1':
            manual_testing(model)

        elif choice == '2':
            batch_testing(model)

        elif choice == '3':
            print(f"\n📋 ИНФОРМАЦИЯ О МОДЕЛИ")
            print(f"  Устройство: {device}")
            print(f"  Длина массива: {ARRAY_LENGTH}")
            print(f"  Диапазон чисел: 1-{MAX_VALUE}")
            print(f"  Размер входа: {INPUT_SIZE}")
            print(f"  Архитектура: {HIDDEN_LAYERS}")
            print(f"  Dropout: {DROPOUT}")

            # Считаем параметры
            dummy_model = SortNet5()
            total_params = sum(p.numel() for p in dummy_model.parameters())
            print(f"  Всего параметров: {total_params:,}")

        elif choice == '4':
            model_files = glob.glob(os.path.join("models", "*.pt"))
            if model_files:
                print(f"\n📚 Доступные модели ({len(model_files)}):")
                for i, f in enumerate(sorted(model_files, key=os.path.getctime, reverse=True)[:10]):
                    size = os.path.getsize(f) / 1024 / 1024  # MB
                    mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
                    print(f"  {i+1:2d}. {os.path.basename(f)}")
                    print(f"      Размер: {size:.2f} MB, Изменен: {mtime}")
            else:
                print("  Нет сохраненных моделей")

        elif choice == '5':
            print("\n📂 Укажите путь к файлу модели")
            custom_path = input("Путь (или Enter для отмены): ").strip()
            if custom_path and os.path.exists(custom_path):
                new_model, _ = load_model(custom_path)
                if new_model is not None:
                    model = new_model
                    print("✅ Модель перезагружена!")
            else:
                print("❌ Файл не найден")

        elif choice == '6':
            print("\n👋 До свидания!")
            break

        else:
            print("❌ Неверный выбор. Введите 1-6")

def quick_test(model):
    """Быстрый тест для демонстрации"""
    print("\n" + "="*70)
    print("⚡ БЫСТРЫЙ ТЕСТ")
    print("="*70)

    test_cases = [
        ([1, 2, 3, 4, 5], "Возрастающий"),
        ([5, 4, 3, 2, 1], "Убывающий"),
        ([42, 42, 42, 42, 42], "Все одинаковые"),
        ([1, 100, 50, 25, 75], "Смешанный"),
        ([23, 67, 12, 89, 45], "Случайный 1"),
        ([98, 99, 100, 97, 96], "Почти max")
    ]

    print(f"\n{'Тест':<15} | {'Вход':<25} | {'Предсказано':<20} | {'Результат'}")
    print("-"*85)

    for test_arr, desc in test_cases:
        predicted = predict(model, test_arr)
        correct = sorted(test_arr)
        is_correct = predicted == correct
        status = "✅" if is_correct else "❌"

        arr_str = str(test_arr)
        pred_str = str(predicted)
        print(f"{desc:<15} | {arr_str:<25} | {pred_str:<20} | {status}")

def main():
    print("="*80)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ ДЛЯ СОРТИРОВКИ 5 ЧИСЕЛ")
    print("="*80)

    # Загружаем модель
    model, checkpoint = load_model()

    if model is None:
        print("\nХотите указать путь к модели вручную?")
        custom_path = input("Путь (или Enter для выхода): ").strip()
        if custom_path and os.path.exists(custom_path):
            model, checkpoint = load_model(custom_path)
        else:
            print("Выход из программы.")
            return

    # Быстрый тест для демонстрации
    quick_test(model)

    # Запускаем меню
    interactive_menu(model)

if __name__ == "__main__":
    main()
