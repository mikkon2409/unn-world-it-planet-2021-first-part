import cv2 as cv


def extract_doc_features(filepath: str) -> dict:
    """
    Функция, которая будет вызвана для получения признаков документа, для которого задан:
    :param filepath: абсолютный путь до тестового файла на локальном компьютере (строго pdf или png).

    :return: возвращаемый словарь, совпадающий по составу и написанию ключей условию задачи
    result = {
        'red_areas_count': int, # количество красных участков (штампы, печати и т.д.) на скане
        'blue_areas_count': int, # количество синих областей (подписи, печати, штампы) на скане
        'text_main_title': str, # текст главного заголовка страницы или ""
        'text_block': str, # текстовый блок параграфа страницы, только первые 10 слов, или ""
        'table_cells_count': int, # уникальное количество ячеек (сумма количеств ячеек одной или более таблиц)
    }
    """

    result_dict = {}
    return result_dict


if __name__ == '__main__':
    import argparse
    import os
    print(os.getenv("INPUTS_FOLDER", default="inputs"))
