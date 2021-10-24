import pytesseract
import cv2 as cv
import re
import numpy as np
import pdf2image


def hard_process(image, minHSV, maxHSV):
    frame_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_hsv, minHSV, maxHSV)

    kernel = np.ones((3, 3), np.uint8)
    frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)
    frame_threshold = cv.dilate(frame_threshold, kernel, iterations=7)

    contours, hierarchy = cv.findContours(frame_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame_threshold, contours, -1, 255, -1)

    contours = [contour for contour in contours if len(contour) > 50]

    return frame_threshold, len(contours)


def detect_box(image, line_min_width=15):
    gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    th1, img_bin = cv.threshold(gray_scale, 150, 225, cv.THRESH_BINARY)
    kernel_h = np.ones((1, line_min_width), np.uint8)
    kernel_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv.morphologyEx(~img_bin, cv.MORPH_OPEN, kernel_h)
    img_bin_v = cv.morphologyEx(~img_bin, cv.MORPH_OPEN, kernel_v)
    img_bin_final = img_bin_h | img_bin_v
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv.dilate(img_bin_final, final_kernel, iterations=1)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv.CV_32S)
    return stats, labels, ret


def repack(dict_of_lists):
    list_of_dicts = []
    length = len(dict_of_lists[list(dict_of_lists.keys())[0]])
    keys = dict_of_lists.keys()
    for i in range(length):
        tmp = {}
        for key in keys:
            tmp[key] = dict_of_lists[key][i]
        list_of_dicts.append(tmp)
    return list_of_dicts


def box_contain_box(big_box, small_box):
    big_left = big_box['left']
    big_top = big_box['top']
    big_right = big_left + big_box['width']
    big_bottom = big_top + big_box['height']
    small_left = small_box['left']
    small_top = small_box['top']
    small_right = small_left + small_box['width']
    small_bottom = small_top + small_box['height']

    return big_left <= small_left < small_right <= big_right and big_top <= small_top < small_bottom <= big_bottom


def boxes_inside_box(big_box, small_boxes):
    small_boxes_inside = []
    for small_box in small_boxes:
        if box_contain_box(big_box, small_box):
            small_boxes_inside.append(small_box)

    return small_boxes_inside


def pdf_or_png(file_path: str):
    with open(file_path, "rb") as file:
        format_bytes = file.read()[1:4]
        if format_bytes == b'PDF':
            return "PDF"
        elif format_bytes == b'PNG':
            return "PNG"


def get_image_from_filepath(filepath: str) -> np.ndarray:
    image = None
    file_type = pdf_or_png(filepath)
    if file_type == "PDF":
        images = pdf2image.convert_from_path(filepath, 300)
        image = cv.cvtColor(np.array(images[0]), cv.COLOR_RGB2BGR)
    elif file_type == "PNG":
        image = cv.imread(filepath)
    return image


def normalize_title(title):
    for i in range(len(title)):
        if title[i]['text'].isupper():
            if i == 0:
                title[i]['text'] = title[i]['text'].capitalize()
            else:
                title[i]['text'] = title[i]['text'].lower()
    return title


def postprocess_of_ocr(list_of_features):
    whole_page_desc, valid_paragraph_boxes, text_boxes = get_valid_boxes(list_of_features)

    title = ocr_title_filter(whole_page_desc, valid_paragraph_boxes, text_boxes)
    title = normalize_title(title)
    if len(title) > 0:
        title = " ".join([x['text'] for x in title])
    else:
        title = ""

    text = ocr_text_filter(whole_page_desc, valid_paragraph_boxes, text_boxes)
    if len(text) > 0:
        text = " ".join([text[i]['text'] for i in range(min(len(text), 10))])
    else:
        text = ""

    return title, text


def get_valid_boxes(list_of_features):
    whole_page_desc = list(filter(lambda x: x['level'] == 1, list_of_features))[0]

    all_big_boxes = list(filter(lambda x: x['level'] == 2, list_of_features))

    text_boxes = list(
        filter(lambda x: int(x['conf']) > 0 and re.match(r'[А-Яа-я]+', x['text']) is not None, list_of_features))
    reg = re.compile('[^а-яА-Я]')
    for item in text_boxes:
        item['text'] = reg.sub('', item['text'])

    big_boxes_with_text = []
    for box in all_big_boxes:
        text_inside = boxes_inside_box(box, text_boxes)
        if len(text_inside) > 0:
            if len(text_inside) == 1:
                box['left'] = text_inside[0]['left']
                box['top'] = text_inside[0]['top']
                box['width'] = text_inside[0]['width']
                box['height'] = text_inside[0]['height']
            big_boxes_with_text.append(box)

    valid_paragraph_boxes = list(
        filter(lambda x: 0 <= len(boxes_inside_box(x, big_boxes_with_text)) <= 1, big_boxes_with_text))

    return whole_page_desc, valid_paragraph_boxes, text_boxes


def get_median_text_size(text_boxes: list):
    boxes = text_boxes.copy()
    boxes.sort(key=lambda x: x['height'])
    return boxes[len(boxes) // 2]['height']


def is_paragraph(text_boxes, page_width) -> bool:
    lines = set()
    paragraphs = set()
    for box in text_boxes:
        lines.add(box['line_num'])
    for box in text_boxes:
        paragraphs.add(box['par_num'])

    for line in lines:
        for par in paragraphs:
            one_line = list(filter(lambda x: x['line_num'] == line and x['par_num'] == par, text_boxes))
            # if len(one_line) < 2:
            #     return False
            one_line.sort(key=lambda x: x['word_num'])
            for i in range(len(one_line) - 1):
                distance = one_line[i + 1]['left'] - (one_line[i]['left'] + one_line[i]['width'])
                if distance > 0.1 * page_width:
                    return False

    return True


def ocr_text_filter(whole_page_desc, big_boxes, word_boxes):
    page_width = whole_page_desc['width']
    median_text_size = get_median_text_size(word_boxes)
    maybe_paragraph = list(
        filter(lambda x:
               x['width'] > 0.7 * page_width and
               x['height'] >= 2 * median_text_size and
               is_paragraph(boxes_inside_box(x, word_boxes), page_width), big_boxes)
    )
    maybe_paragraph.sort(key=lambda x: x['top'])

    text = []
    if len(maybe_paragraph) > 0:
        text_box = maybe_paragraph[0]
        text = boxes_inside_box(text_box, word_boxes)

    return text


def ocr_title_filter(whole_page_desc, big_boxes, word_boxes):
    page_width = whole_page_desc['width']
    page_horizontal_center = page_width / 2

    def center_of_box(box):
        return abs(box['left'] + box['width'] / 2)

    maybe_title = list(
        filter(lambda x: (center_of_box(x) - page_horizontal_center) < 0.1 * page_width, big_boxes))
    maybe_title.sort(key=lambda x: x['top'], reverse=True)
    for i in range(len(maybe_title)):
        maybe_title[i]['score'] = i / len(maybe_title)

    maybe_title.sort(key=lambda x: x['score'], reverse=True)
    title_box = maybe_title[0]
    title = boxes_inside_box(title_box, word_boxes)
    title = list(filter(lambda x: x['line_num'] == 1, title))
    title.sort(key=lambda x: x['word_num'])

    return title


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

    img_src = get_image_from_filepath(filepath)

    features = pytesseract.image_to_data(img_src, lang="rus", output_type=pytesseract.Output.DICT)
    features = repack(features)

    title, text = postprocess_of_ocr(features)

    _, blue_contour_count = hard_process(img_src, (53, 35, 134), (150, 255, 255))

    _, red_contour_count = hard_process(img_src, (155, 64, 94), (210, 250, 249))

    # _, _, ret = detect_box(img_src)

    result_dict = {
        'red_areas_count': red_contour_count,
        'blue_areas_count': blue_contour_count,
        'text_main_title': title,
        'text_block': text,
        'table_cells_count': 0
    }
    return result_dict
