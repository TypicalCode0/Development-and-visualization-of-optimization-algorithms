# Development-and-visualization-of-optimization-algorithms

## Оглавление

- [Оглавление](#оглавление)
- [1 Вступление](#1-вступление)
- [2 Запуск программы](#2-запуск-программы)
- [3 UI](#3-ui)

***

## 1 Вступление

Цель данной курсовой работы - визуализация работы алгоритмов оптимизации(пока в процессе), таких как Gradient Descent,
Sequential Programming, Interior-point methods.  
Работа выполняется: TypicalCode0(Аладышев Дмитрий), SlowpokerFace(Шестиперстов Валентин), AnixGG(Газизулин Нияз),
LeTim42(Лебедев Тимур)
Научный руководитель: Бычков Илья Сергеевич

## 2 Запуск программы

1. Клонирование репозитория
2. Установить python, если ещё это не сделали
3. Установить нужные библиотеки: `pip install requirements.txt`
4. Скомпилировать бинарные файлы через CMake:
   - `cmake -B bin cpp` - конфигурация
   - `cmake --build bin --config Release` - сборка
5. Запустить setup.py
6. Правила заполнения полей:
   - поддерживаемые операции в функции: + - * ^(**) / ( )
   - limitations - ограничения, можно задавать несколько, разделяя знаком ";"
   - universe - задание левой и правой границ для всех переменных

> [!NOTE]
> Возможно в будущем сделаем makefile

## 3 UI

Ссылка на figma с концептом UI https://www.figma.com/file/6peWaiucRqFpiL5yhAFPFg/UI-concept?type=design&node-id=0-1&mode=design&t=qiihIj0c59pyGC1b-0
