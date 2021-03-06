# Проект "Определение категорий новостных текстов"
Выполнен студентками 3 курса Николаенковой Марией, Моховой Анной и Александровой Полиной.

Проект направлен на создание приложения, в котором пользователь может ввести собственный текст новости и узнать, к какой категории она относится. Обучение классификатора производилось на материале текстов с новостного портала Lenta.ru (объём корпуса - 50 000 статей). 

Датасет можно скачать по [ссылке](https://drive.google.com/file/d/1-6ECrlJB69HMTB1M3W1P6BwHVeCS0hLY/view?usp=sharing).

Репозиторий состоит из следующих файлов:
* [README.md](https://github.com/nikolmash/news_category_detection/blob/master/README.md) - описание проекта и пререквизитов.
* [data_preproc.ipynb](https://github.com/nikolmash/news_category_detection/blob/master/data_preproc.ipynb) - ноутбук, подготавливающий датасет для обучения (включает в себя очистку пустых данных, анализ распределения классов и исключение малочисленных категорий)
* [learning.py](https://github.com/nikolmash/news_category_detection/blob/master/learning.py) - скрипт, реализующий непосредственно обучение (в результате сохраняет необходимые веса для TF_IDF и непосредственно самой модели логистической регрессии)
* [get_models.py](https://github.com/nikolmash/news_category_detection/blob/master/get_models.py) - скрипт, реализующий загрузку готовых файлов с моделями (включает в себя проверку на существование файлов в директории для исключения лишней загрузки).
* [preprocessing.py](https://github.com/nikolmash/news_category_detection/blob/master/preprocessing.py) - скрипт, реализующий препроцессинг текста, введённого пользователем.
* [app.py](https://github.com/nikolmash/news_category_detection/blob/master/app.py) - непосредственно запуск самого приложения с необходимыми реквизитами:
  * [static](https://github.com/nikolmash/news_category_detection/tree/master/static)
  * [templates](https://github.com/nikolmash/news_category_detection/tree/master/templates)
* [wordcloud_example.ipynb](https://github.com/nikolmash/news_category_detection/blob/master/wordcloud_example.ipynb) - ноутбук с функцией, создающей облако из слов, наиболее важных для классфикации. Облака для моделей с различными настройками обработки текста доступны по ссылке с главной страницы.

## Пререквизиты
* Перед началом работы с приложением настоятельно рекомендуется запустить force_download из get_models.py для того, чтобы заранее сохранить файлы с моделями, которые потребуются для предсказания категории.
* Для запуска приложения необходима предустановка следующих библиотек:
  * nltk 3.5
  * pymorphy2 0.8
  * Flask 1.1.2
  * matplotlib 3.2.2

## Работа приложения
Помимо ввода новостоного текста пользователь может выбрать, какие из этапов обработи текстов стоит применять для его новости (исключение стоп-слов и лемматизация текста).

В результате классфикации введённой новости пользователю предлагается оценить правильность предсказания. Все ответы пользователей сохраняются в базу данных data.db для дальнейшей статистики правильности результатов. Пример визулизации правильности классификатора можно посмотреть [здесь](https://github.com/nikolmash/news_category_detection/blob/master/static/statistics.png).
