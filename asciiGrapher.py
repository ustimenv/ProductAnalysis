import psycopg2
import datetime
import math
import smtplib
from email.mime.text import MIMEText
from email.header import Header


class ParamsAndConstants:
    """"
    Для всяких параметров подключения и констант, необходимых для работы подсистемы
    """

    def __init__(self):
        # параметры подключения к базе данных (Postgres)
        self.sqlparams = ["10.110.0.46", "5432", "ConveyorData", "conveyoradm", "Ax321tcw7#5Y"]
        # с этой даты загружаем данные для камеры (сдвиг ** дней от сегодня и ** часов)
        date_load_from = datetime.datetime.now().replace(hour=15,
                                                         minute=50,
                                                         second=0,
                                                         microsecond=0) + datetime.timedelta(days=-1)
        # загружаем даные до этого времени (типа до сегодня, до 15 часов или до полуночи).
        date_load_to = datetime.datetime.now().replace(hour=23,
                                                       minute=59,
                                                       second=0,
                                                       microsecond=0) + datetime.timedelta(days=-0)
        self.date_load_from_str = date_load_from.strftime("%Y-%m-%d %H:%M")  # чтобы сто раз не конвертировать
        self.date_load_to_str = date_load_to.strftime("%Y-%m-%d %H:%M")  # чтобы сто раз не конвертировать
        self.camera_off_delay_minutes = 3
        self.no_product_delay_minutes = 6


def day_to_str(in_day):
    """"
    Вспомогательная функция, чтобы можно вывести 23 февраЛЯ (а не февраль)
    на входе: datetime, на выходе - строка '23 февраля'
    """
    monthes_list = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
    result_str = f"{in_day.day} {monthes_list[in_day.month-1]}"
    return result_str


def status_in_ru(status_eng):
    """"
    Вспомогательная функция, выдаёт строку статуса на русском для отчета
    """
    statuses_in_ru = {'start': 'начало работы', 'counting..': 'считаем продукцию..',
                      'no_products_found': 'считаем, но нет продукции',
                      'camera_off': 'камера выключена', 'finish': 'окончание работы', 'job_status': 'job_status'}
    result_str = f"{statuses_in_ru[status_eng]}"
    return result_str


def interval_from_to(dt_from, dt_to):
    """"
    Вспомогательная функция, выдаёт строку временного интервала С _ ПО _ (типа - красиво и компактно)
    """
    if dt_from.date() == dt_to.date():
        result_str = '' + f"{dt_from.strftime('%Y-%m-%d ')} c {dt_from.strftime('%H:%M')} по {dt_to.strftime('%H:%M')}"
    else:
        result_str = '' + f"с {dt_from.strftime('%Y-%m-%d %H:%M')} по {dt_to.strftime('%Y-%m-%d %H:%M')}"
    return result_str


def interval_duration(dt_from, dt_to):
    """"
    Вспомогательная функция, выдаёт строку временного интервала С _ ПО _ (типа - красиво и компактно)
    """
    if dt_from == dt_to:
        result_str = ''
    else:
        time_delta = dt_to - dt_from
        result_str = '('
        # result_str = f'( *{time_delta.seconds // (60*60)}*'
        if int(time_delta.seconds // (60 * 60)) > 0:
            result_str = result_str + f"{time_delta.seconds // (60*60)} час "
        result_str = result_str + f"{(time_delta.seconds//60)%60} мин"

        result_str = result_str + ')'
    return result_str


def int_to_str(in_digit):
    """"
    Вспомогательная функция, переводит целое число в строку (типа - красиво и компактно) вида 12'005'007
    """
    m = False
    if in_digit < 0:
        result_str = '-'
        in_digit = - in_digit
    else:
        result_str = ''
    if in_digit < 1000:
        result_str = result_str + f"{int(round(in_digit))}"
    else:
        if in_digit > 1000000:
            result_str = result_str + f"{int(in_digit/1000000)}'"
            in_digit = in_digit - int(in_digit / 1000000) * 1000000
            m = True
        if m:
            result_str = result_str + f"{int(in_digit/1000):03d}'"
            in_digit = in_digit - int(in_digit / 1000) * 1000
        elif in_digit > 1000:
            result_str = result_str + f"{int(in_digit/1000)}'"
            in_digit = in_digit - int(in_digit / 1000) * 1000
        result_str = result_str + f"{int(round(in_digit)):03d}"
    return result_str


class EmailHTML:
    def __init__(self):
        self.smtpObj = smtplib.SMTP('smtp.darnitsa.ru', 25)  # 25  587
        self.smtpObj.ehlo()
        self.smtpObj.starttls()
        self.smtpObj.ehlo()
        self.smtpObj.login('R_Robot@darnitsa.ru', 'FGp2851swD#96!')  # login & pass
        self.from_addr = "R_Robot@darnitsa.ru"
        # self.to = ["nbmikhelson@darnitsa.ru",  "mikolya@gmail.com"]
        # self.to = ["nbmikhelson@darnitsa.ru"]

        self.to = ["VAUstimenko@darnitsa.ru"]
        self.to_str = ["По списку рассылки.. "]
        self.to_str0 = "По списку рассылки.. "
        for addr in self.to:
            self.to_str0 = self.to_str0 + ' ' + addr
        self.to_str = [self.to_str0]
        #
        # инициализируем header & footer
        self.html_header = '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN"><html>'
        self.html_header = self.html_header + '<head content="text/html" http-equiv="Content-Type" charset="utf-8">'
        self.html_header = self.html_header + '<meta http-equiv="Content-Type" content="text/html; ' \
                                              'charset=utf-8"></head>'
        self.html_header = self.html_header + '<body>'
        self.html_footer = '</body></html></html>'
        self.html_greeting = '<p><font color="#536ac2">Дорогие <strike>мои</strike> Коллеги!</font></p>'
        self.html_greeting = self.html_greeting + '<p><font color="#800000">Вот, полюбуйтесь, ' \
                                                  'как работала камера машинного зрения на линии 3..' \
                                                  '</font></p><p><br></p>'
        self.html_adieu = '<p><br></p><p><font color="#536ac2">С Уважением, </font></p>'
        self.html_adieu = self.html_adieu + '<blockquote style="MARGIN-RIGHT: 0px" dir="ltr">'
        self.html_adieu = self.html_adieu + '<blockquote style="MARGIN-RIGHT: 0px" dir="ltr">'
        self.html_adieu = self.html_adieu + '<p><font color="#536ac2"><strike>Ваш любимый Робот</strike>'
        self.html_adieu = self.html_adieu + ' Система Информационного оповещения ГК Дарница' \
                                            '</font></p></blockquote></blockquote>'
        self.message_html = None
        self.msg = None

    def get_message_html(self, in_message_html):
        self.message_html = self.html_header + self.html_greeting + in_message_html + self.html_adieu + self.html_footer
        self.msg = MIMEText(self.message_html, 'html', 'utf-8')
        self.msg['Subject'] = Header(
            f'Отчет Робота о работе камеры машинного зрения на линии 3 по {day_to_str(datetime.datetime.now())} '
            f'{datetime.datetime.now().year}',
            'utf-8')
        self.msg['From'] = self.from_addr
        self.msg['To'] = ", ".join(self.to_str)

    def send_email(self):
        self.smtpObj.sendmail(self.from_addr, self.to, self.msg.as_string())
        # self.msg = ''


class CameraData:
    def __init__(self):
        self.parameters = ParamsAndConstants()
        """
        основной первичный массив данных, загруженных из SQL храним в rowdata
        rowdata = список, каждый элемент которого = словарь из времени регистрации и количества хлебов
        """
        self.rowdata = []  # self.rowdata = [{'dtime':0,'num_objects':0}]
        self.rowdata_counter = 0  # на всякий случай - количество загруженных записей для контроля (д.б. = len(rowdata))
        self.load_cam_log()  # загружаем данные из SQL
        self.report_working_time = []  # тут удет отчета о работе (подсчете) за период (описание структуры - ниже)
        self.report_working_time_build()  # формируем отчет о работе (подсчете) за период
        #
        # self.vals_in_minute[]  -  в этот список поместим пересчитанные на минуту результаты измерений.
        # self.vals_in_minute[ 30 ]  =  40  - это значит, что 40 раз насчитали 30 хлебов в минуту
        self.vals_in_minute = []
        self.val_minute_max = 0  # максимум было вот столько хлебов в минуту
        self.val_minute_min = 0  # минимум было вот столько хлебов в минуту (очень мало, но не 0)
        self.dispersia = 0
        self.median = 0
        self.dispersia_korrected = 0
        self.average_weighted = 0
        self.sigma = 0
        self.moda = 0
        self.asymetry_moment = 0  # моментный коэффициент асимметрии
        self.asymetry_moment_sqr_err = 0  # средняя квадратичная ошибка коэффициента асимметрии

    def load_cam_log(self):
        """"
        Загружаем данные из базы данных
        все посчеты за период (от даты date_load_from)
        загружаем время регистрации (записи: sample_time) и количество подсчитанных хлебов (num_objects)
        """
        query_select = f"select sample_time, num_objects from conveyor_count where line = 1 and position = 1" \
                       f"and sample_time >= '{self.parameters.date_load_from_str}' " \
                       f"and sample_time <= '{self.parameters.date_load_to_str}' " \
                       f"order by sample_time asc"
        # print(query_select)
        conn3 = psycopg2.connect(dbname=self.parameters.sqlparams[2], user=self.parameters.sqlparams[3],
                                 password=self.parameters.sqlparams[4], host=self.parameters.sqlparams[0],
                                 port=self.parameters.sqlparams[1])
        cursor = conn3.cursor()
        cursor.execute(query_select)
        self.rowdata_counter = 0
        for row in cursor:
            sample_time = row[0]  # - лишние строки, можно и без них
            num_objects = row[1]  # - но с ними - нагляднее, что делаем.
            event = {'dtime': sample_time, 'num_objects': num_objects}
            self.rowdata.append(event)  # добавили событие в список событий камеры
            self.rowdata_counter += 1

    def report_working_time_build(self):
        """"
        формируем отчетик в виде списка (list),
        каждый элемент которого - временной интервал с разными статусами режима в виде словаря, в каждом элементе:
                        номер инрервала, дата_время с_,  дата_время по_,
                        job_status {начали, считаем и есть продукция, считаем нет продукции, не считаем, закончили}
                        количество подсчитаных объектов (хлебов)
        """
        interval = 0
        # первое событие - начало работы, фиксируем его событием в отчет.
        event = self.rowdata[0]
        event_interval_0 = {'interval': interval, 'time_from': event['dtime'], 'time_to': event['dtime'],
                            'job_status': 'start', 'num_objects': 0}
        interval += 1
        time0 = event['dtime']
        time_with_out_products_from = event['dtime']
        event_interval_00 = []
        job_status0 = 'start'

        # бежим в цикле по всем событиям - строкам rowdata с камеры..
        for event in self.rowdata:
            # нет событий от камеры больше 3-х минут - считаем, что камера была отключена   :(
            if event['dtime'] - event_interval_0['time_to'] > \
                    datetime.timedelta(minutes=self.parameters.camera_off_delay_minutes):
                # не было событий от камеры больше 3-х минут - считаем, что камера была отключена   :(
                event_report = {'interval': interval, 'time_from': time0, 'time_to': event['dtime'],
                                'job_status': 'camera_off', 'num_objects': 0}
                # print('--- 1+1 ---')
                self.report_working_time.append(event_interval_0)
                # self.report_working_time.append(event_report)
                event_interval_0 = event_report
                interval += 1
                job_status0 = 'camera_off'

            if event['num_objects'] == 0:
                job_status = 'no_products_found'
            else:
                job_status = 'counting..'
            # Job status не изменился, продолжаем считать (сделано)
            if job_status0 == job_status == 'counting..':
                """
                Job status не изменился, продолжаем считать и -> расширяем интервал на текущее событие
                """
                event_interval_0 = {'interval': event_interval_0['interval'],
                                    'time_from': event_interval_0['time_from'],
                                    'time_to': event['dtime'],
                                    'job_status': event_interval_0['job_status'],
                                    'num_objects': event_interval_0['num_objects'] + event['num_objects']}
            # Job status изменился, пошла продукция и начали считать
            elif job_status0 == 'no_products_found' and job_status == 'counting..':
                """
                Job status изменился, пошла продукция..  
                если до этого небыло продукции до трёх минут, то простой без продукции не считается простоем
                """
                if event['dtime'] - time_with_out_products_from < \
                        datetime.timedelta(minutes=self.parameters.no_product_delay_minutes):
                    # было до трёх минут без продукции, удаляем старый без_продуктовый интервал
                    # вернее, тот безпродуктовый интервал переделываем в подсчет
                    # продолжаем старый не записанный интервал..
                    event_interval_0 = event_interval_00
                    interval -= 1
                    event_interval_0 = {'interval': event_interval_0['interval'],
                                        'time_from': event_interval_0['time_from'],
                                        'time_to': event['dtime'],
                                        'job_status': job_status,
                                        'num_objects': event_interval_0['num_objects'] + event['num_objects']}
                    job_status0 = job_status
                else:
                    # был простой без продукции больше 3-х минут, записываем простой и создаём новый интервал
                    # print('--- 2+2 ---')
                    if event_interval_00:
                        # print(event_interval_00)
                        self.report_working_time.append(event_interval_00)
                    # print(event_interval_0)
                    self.report_working_time.append(event_interval_0)
                    event_interval_0 = {'interval': interval,
                                        'time_from': event['dtime'],
                                        'time_to': event['dtime'],
                                        'job_status': job_status,
                                        'num_objects': event['num_objects']}
                    interval += 1  # начали новый интервал
                    job_status0 = job_status
            # работали, но теперь продукции нет
            elif job_status0 == 'counting..' and job_status == 'no_products_found':
                time_with_out_products_from = event_interval_0['time_to']
                event_interval_00 = event_interval_0  # ivent_interval 00 пока не записываем, вдруг это не простой линии
                # создаём новый интервал без продукции..
                event_interval_0 = {'interval': interval,
                                    'time_from': event['dtime'],
                                    'time_to': event['dtime'],
                                    'job_status': job_status,
                                    'num_objects': event['num_objects']}
                interval += 1  # начали новый интервал
                job_status0 = job_status
            # продукции до сих пор нет
            elif job_status0 == job_status == 'no_products_found':
                # продукции до сих пор нет, расширяем интервал без продукции
                event_interval_0 = {'interval': event_interval_0['interval'],
                                    'time_from': event_interval_0['time_from'],
                                    'time_to': event['dtime'],
                                    'job_status': job_status,
                                    'num_objects': event_interval_0['num_objects'] + event['num_objects']}
            elif job_status0 != job_status:  # and job_status == 'counting..':
                # записываем старый интервал и создаём новый.
                # print('job_status0 != job_status', job_status, event_interval_0)
                # print('--- 3 ---', event_interval_0)
                self.report_working_time.append(event_interval_0)
                event_interval_0 = {'interval': interval,
                                    'time_from': event['dtime'],
                                    'time_to': event['dtime'],
                                    'job_status': job_status,
                                    'num_objects': event['num_objects']}
                event_interval_00 = []
                interval += 1
                job_status0 = job_status
            else:
                # статус не изменился -> расширяем интервал на текущее событие
                event_interval_0 = {'interval': event_interval_0['interval'],
                                    'time_from': event_interval_0['time_from'],
                                    'time_to': event['dtime'],
                                    'job_status': event_interval_0['job_status'],
                                    'num_objects': event_interval_0['num_objects'] + event['num_objects']}
            time0 = event['dtime']

        # последнее событие - окончание работы, фиксируем его событием в отчет.
        event_report = {'interval': interval, 'time_from': event['dtime'], 'time_to': event['dtime'],
                        'job_status': 'finish', 'num_objects': 0}
        # записываем прошлый интервал (уже закончившийся)
        # print(event_interval_0)
        # print('--- 4 ---', event_interval_0)
        if event_interval_00:
            # print(event_interval_00)
            self.report_working_time.append(event_interval_00)
        self.report_working_time.append(event_interval_0)
        # и записываем финальное событие
        # print('--- 5 ---', event_report)
        self.report_working_time.append(event_report)

        interval += 1

    def report_working_time_print(self):
        result_str = ''
        for interval in self.report_working_time:
            report_str = f"{interval['interval']} c {interval['time_from'].strftime('%Y-%m-%d %H:%M')} " \
                         f"по {interval['time_to'].strftime('%Y-%m-%d %H:%M')}, " \
                         f"{interval['job_status']}"
            print(report_str)
            result_str = result_str + '\n' + report_str
        return result_str

    def report_working_time_ru_print(self):
        result_str = ''
        total_quantity = 0
        for interval in self.report_working_time:
            quantity_str = ''
            if interval['job_status'] == 'counting..':
                quantity_str = f" {int_to_str(interval['num_objects'])} шт"
                total_quantity += interval['num_objects']
            report_str = f"{interval_from_to(interval['time_from'], interval['time_to'])}" \
                         f" {interval_duration(interval['time_from'], interval['time_to'])}" \
                         f": {status_in_ru(interval['job_status'])}" + quantity_str
            print(report_str)
            result_str = result_str + '\n' + report_str
        report_str = f"\nВсего с {self.report_working_time[0]['time_from'].strftime('%Y-%m-%d %H:%M')} " \
                     f"по {self.report_working_time[-1]['time_to'].strftime('%Y-%m-%d %H:%M')} " \
                     f"насчитали {int_to_str(total_quantity)} шт."
        print(report_str)
        result_str = result_str + '\n' + report_str
        return result_str

    def report_working_time_html(self):
        """
        Возвращаем в виде HTML (только блок <body> без заголовка)
        отчет по интервалам (с__ по__ длительность__ статус интервала__ количество__ )
        и выводим итоговую строку:  Всего с *** по *** насчитали **'*** шт
        """
        result_html = '<p><strong><font color="#ff0066">=== Отчет по интервалам ===</strong></font></p><p></p><p>'
        total_quantity = 0
        for interval in self.report_working_time:
            quantity_str = ''
            if interval['job_status'] == 'counting..':
                quantity_str = f" {int_to_str(interval['num_objects'])} шт"
                total_quantity += interval['num_objects']
            # определяем цвета исходя из статуса интервала
            if interval['job_status'] == 'counting..' or interval['job_status'] == 'finish' or interval[
                                         'job_status'] == 'start':
                color_from_to = '#006400'  # зеленый
                color_duration = '#006400'
                color_status = '#006400'
                if interval['num_objects'] < 5:
                    color_num_objects = '#ff0000'  # красный
                else:
                    color_num_objects = '#006400'  # зеленый
            elif interval['job_status'] == 'no_products_found':
                color_from_to = '#000080'  # синий
                color_duration = '#000080'
                color_status = '#000080'
                if interval['num_objects'] < 5:
                    color_num_objects = '#ff0000'  # красный
                else:
                    color_num_objects = '#000080'  # зеленый
            elif interval['job_status'] == 'camera_off':
                color_from_to = '#ff0000'  # красный
                color_duration = '#ff0000'
                color_status = '#ff0000'
                color_num_objects = '#ff0000'
            else:
                color_from_to = '#A9A9A9'  # серый
                color_duration = '#A9A9A9'
                color_status = '#A9A9A9'
                color_num_objects = '#A9A9A9'

            report_row = f"" \
                         f'<font color="{color_from_to}">' \
                         f"{interval_from_to(interval['time_from'], interval['time_to'])}</font>" \
                         f'<font color="{color_duration}">' \
                         f" {interval_duration(interval['time_from'], interval['time_to'])}</font>" \
                         f'<font color="{color_status}">' \
                         f": {status_in_ru(interval['job_status'])}</font>" \
                         f'<font color="{color_num_objects}">{quantity_str}</font>' \
                         f""
            # print(report_row)
            result_html = result_html + '<br>' + report_row
        report_row = f'<p><br><strong>Всего <font color="#000080">с ' \
                     f"{self.report_working_time[0]['time_from'].strftime('%Y-%m-%d %H:%M')} " \
                     f"по {self.report_working_time[-1]['time_to'].strftime('%Y-%m-%d %H:%M')} </font></strong>" \
                     f'насчитали <strong><font color="#006400">{int_to_str(total_quantity)}</font></strong> шт.</p>'
        # print(report_row)
        result_html = result_html + '<br>' + report_row
        return result_html

    def report_row_data_print(self):
        for event in self.rowdata:
            report_str = f"{event['dtime'].strftime('%Y-%m-%d %H:%M')} " \
                         f" количество: {event['num_objects']}"
            print(report_str)
        pass

    def calc_stat_1_values_list(self):
        # формируем временный список измерений, приведенный в штукам в минуту
        # self.vals_in_minute[]  -  в этот список поместим пересчитанные на минуту результаты измерений.
        # self.vals_in_minute[ 30 ]  =  40  - это значит, что 40 раз насчитали 30 хлебов в минуту
        row_data_by_minutes = []
        data_rows = len(self.rowdata)
        self.val_minute_max = 0
        self.val_minute_min = 0
        for data_i in range(0, data_rows - 1):
            # Бежим в цикле по всем событиям и везде, переводим посчитанное камерой за период в хлеба в минуту
            # да, происходит округление до целого, поэтому данные не для экономики, а оценки статистики распределения
            if self.rowdata[data_i]['num_objects'] > 0:
                timedelta = self.rowdata[data_i]['dtime'] - self.rowdata[data_i - 1]['dtime']
                # хлебов в минуту
                pieces_in_minute = int(self.rowdata[data_i]['num_objects'] * 60 / timedelta.seconds)
                row_data_by_minutes.append(pieces_in_minute)
                # заодно, считаем максимальное количество хлебов в минуту (чтобы два раза не бегать по массиву / листу)
                if self.val_minute_max < pieces_in_minute:
                    self.val_minute_max = pieces_in_minute
        # создаём лист с результатами в минуту (и зануляем элементы)
        self.vals_in_minute = list(range(0, self.val_minute_max + 1))
        for i in range(1, len(self.vals_in_minute)):
            self.vals_in_minute[i] = 0

        for value_in_minute in row_data_by_minutes:
            if value_in_minute > 0:
                self.vals_in_minute[value_in_minute] += 1
        # собственно, сделали то, что хотели, построили список (массив) у которого индекс - количество хлебов в минуту
        # значение - количество таких минут (подсчетов), у которых насчитали столько хлебов в минуту
        #
        # на всякий случай, заполним сразу минимальный элемент
        self.val_minute_min = 0
        is_set = False
        for val in self.vals_in_minute:
            if not is_set and val > 0:
                self.val_minute_min = val
                is_set = True
        # на всякий случай приберём мусор
        row_data_by_minutes.clear()

    def calc_stat_2_distribution_center(self):
        # распределение: self.vals_in_minute - лист, номер элемента - хлебов в минуту = х, значение - количество раз = у
        # считаем средневзвешенную: average_weighted = sum( x * y ) / sum ( y )
        # также считаем Моду и Медиану распределения
        fi_sum = 0  # f = значение функции распределения, т.е. количество подсчетов
        xfi_sum = 0  # sum (х * f)   х - аргумент функции, количество хлебов в минуту
        max_val = 0  # максимальное количество подсчитанных раз - высота пика, т.е. f_max
        self.average_weighted = 0  # средневзвешенное значение
        self.moda = 0  # мода распределения
        self.median = 0  # медиана распределения
        for i in range(0, len(self.vals_in_minute)):
            fi_sum = fi_sum + self.vals_in_minute[i]
            xfi_sum = xfi_sum + i * self.vals_in_minute[i]
            if self.vals_in_minute[i] > max_val:
                max_val = self.vals_in_minute[i]
                self.moda = i
        fi_sum_05 = fi_sum / 2
        if fi_sum != 0:
            self.average_weighted = xfi_sum / fi_sum
        else:
            self.average_weighted = 999999
        # считаем медиану
        fi_sum_m = 0
        for i in range(1, len(self.vals_in_minute)):
            fi_sum_m += self.vals_in_minute[i]
            if self.median == 0 and fi_sum_m >= fi_sum_05:
                self.median = i

    def calc_stat_3_variation(self):
        """
        распределение: self.vals_in_minute - лист, номер элемента - хлебов в минуту = х, значение - количество раз = у
        считаем показатели вариации:
        дисперсию, сигму, размах
        """
        fi_sum_d = 0
        x_i_minus_x_avg = 0
        for i in range(1, len(self.vals_in_minute)):
            fi_sum_d += self.vals_in_minute[i]
            x_i_minus_x_avg = x_i_minus_x_avg + (i - self.average_weighted) * \
                                                (i - self.average_weighted) * self.vals_in_minute[i]
        self.dispersia = x_i_minus_x_avg / fi_sum_d
        self.dispersia_korrected = x_i_minus_x_avg / (fi_sum_d - 1)
        self.sigma = math.sqrt(self.dispersia)
        m_3 = 0
        for i in range(1, len(self.vals_in_minute)):
            m_3 += ((i - self.average_weighted) * (i - self.average_weighted) * (i - self.average_weighted)) \
                   * self.vals_in_minute[i]
        self.asymetry_moment = (m_3 / fi_sum_d) / (self.sigma * self.sigma * self.sigma)
        self.asymetry_moment_sqr_err = math.sqrt(6 * (self.val_minute_max - 2) /
                                                 ((self.val_minute_max + 1) * (self.val_minute_max + 3)))

    def report_distribution_stat_html(self):
        res_html = ''
        res_html = res_html + '<p><br><strong><font color="#ff0066">=== Показатели центра распределения ===' \
                              '</strong></font></p><p></p>'

        res_html = res_html + f'<p>Средняя взвешенная (выборочная средняя) = ' \
                              f'<strong><font color="#006400">{self.average_weighted:.2f}</font></strong></p>'
        res_html = res_html + f'<p>Мода = <strong><font color="#006400">{self.moda}</font></strong><br>'
        res_html = res_html + f'Мода - наиболее часто встречающееся значение (хлебов в минуту): ' \
                              f'<strong><font color="#006400">{self.moda} </font></strong></p>'

        res_html = res_html + f'<p>Медиана = <strong><font color="#006400">{self.median}</font></strong><br>'
        res_html = res_html + f'Медиана - значение хлебов в минуту, приходящееся на середину выборки: ' \
                              f'в среднем <strong><font color="#006400">{self.median}</font></strong> ' \
                              f'хлебов в минуту<br>'
        res_html = res_html + f'Медиана служит хорошей характеристикой при ассиметричном распределении данных, ' \
                              f'т.к. даже при наличии выбросов данных, медиана более устойчива ' \
                              f'к воздействию отклоняющихся данных.</p>'

        res_html = res_html + f'<p>В симметричных рядах распределения значение моды и медианы совпадают ' \
                              f'со средней величиной <font color="#000080">(xср=Meдиана=Moда)</font><br>, ' \
                              f'а в умеренно асимметричных они соотносятся таким образом: ' \
                              f'<font color="#000080"> <strong>3</strong>(Хср-Meдиана) ≈ Хср-Moда<br> </font>'
        k_ratio = math.fabs((self.average_weighted - self.moda) / (self.average_weighted - self.median))
        if self.average_weighted == self.moda == self.median:
            res_html = res_html + 'В нашем случае выполняется (xср=Meдиана=Moда) и следовательно, ' \
                                  '<font color="#006400">распределение симметрично. Отлично.</font>'
        elif k_ratio < 1:
            res_html = res_html + 'В нашем случае, соотношение <font color="#000080">(Хср-Moда)/(Хср-Meдиана)</font>=' \
                                  f'{k_ratio:.2f} ' \
                                  f'<strong><font color="#006400"> < 1</font></strong> следовательно, ' \
                                  f'распределение является ' \
                                  f'<strong><font color="#006400">почти симметричным </font><strong>'
        elif k_ratio < 3:
            res_html = res_html + 'В нашем случае, соотношение <font color="#000080">(Хср-Moда)/(Хср-Meдиана)</font>=' \
                                  f'{k_ratio:.2f} ' \
                                  f'<strong> <font color="#000080">< 3</font></strong>, ' \
                                  f'следовательно, распределение является <strong><font color="#006400">' \
                                  f'умеренно асимметричным</font></strong> ' \
                                  f'с точки зрения центров распределения (не формы).'
        else:
            res_html = res_html + f'В нашем случае, соотношение <font color="#000080">' \
                                  f'(Хср-Moда)/(Хср-Meдиана)</font>=' \
                                  f'{k_ratio:.2f} ' \
                                  f'<strong><font color="#000080">> 3</font></strong>, ' \
                                  f'следовательно, распределение является <strong><font color="#ff0000">' \
                                  f'асимметричным</font></strong>'
        res_html = res_html + '</p>'

        res_html = res_html + '<p><br><strong><font color="#ff0066">=== Показатели вариации ===' \
                              '</strong></font></p><p></p>'
        res_html = res_html + '<p><font color="#000080">-- Абсолютные показатели вариации. --</font></p><p></p>'
        res_html = res_html + '<p>Размах вариации = мах_х - мин_х  - вроде как нам он прямо сейчас еще не нужен</p>'

        res_html = res_html + f'<p>Дисперсия.. = <strong><font color="#006400">{self.dispersia:.2f}</font></strong><br>'
        res_html = res_html + f'Дисперсия - характеризует меру разброса около ее среднего значения ' \
                              f'(мера рассеивания, т.е. отклонения от среднего). <p>'
        res_html = res_html + f'<p>Дисперсия испр = <strong><font color="#006400">{self.dispersia:.2f}' \
                              f'</font></strong><br>'
        res_html = res_html + f'исправленная дисперсия.. несмещенная оценка дисперсии.. ' \
                              f'состоятельная оценка дисперсии (S^2).<p>'
        res_html = res_html + f'<p><strong><font color="#000080">Сигма</font></strong>.. = ' \
                              f'<strong><font color="#006400">{self.sigma:.2f}</font></strong><br>'
        res_html = res_html + f'3 * Сигма = <font color="#006400"><strong>{3 * self.sigma:.2f}</strong></font>,<br>' \
                              f'[-3 сигма ; среднее ; + 3 сигма] = ' \
                              f'[<font color="#006400">{self.average_weighted - 3*self.sigma:.2f} ; ' \
                              f'{self.average_weighted:.2f} ; <strong>{self.average_weighted + 3*self.sigma:.2f}' \
                              f'</strong></font>]</p>'

        res_html = res_html + '<p><br><strong><font color="#ff0066">=== Показатели формы распределения ===' \
                              '</strong></font></p><p></p>'
        res_html = res_html + '<p>Наиболее точным и распространенным показателем асимметрии является ' \
                              'моментный коэффициент асимметрии. <br>'
        res_html = res_html + '<font color="#000080">As = M_3 / s^3</font> (центральный момент третьего порядка, ' \
                              'деленный на куб среднеквадратического отклонения)<br>'
        res_html = res_html + f'As = <strong><font color="#006400">{self.asymetry_moment:.02f} </strong></font>'
        if self.asymetry_moment < 0:
            res_html = res_html + f' т.к. As меньше нуля (отрицательное), то асимметрия ' \
                                  f'<font color="#000080"><strong>левосторонняя.</strong></font>'
        res_html = res_html + f'<p>'
        res_html = res_html + f'<p>Средняя квадратичная ошибка коэффициента асимметрии:  ' \
                              f'<font color="#000080"><strong>SAs</strong> = sqrt( 6*(n-2) / (n+2)(n+3) ) = ' \
                              f'</font><strong><font color="#006400">{self.asymetry_moment_sqr_err:.02f}' \
                              f'</strong></font><p>'
        res_html = res_html + f'Если выполняется соотношение <font color="#006400">|As|/sAs < 3</font>, ' \
                              f'то асимметрия <font color="#006400">несущественная</font>.<br>'
        asy_ratio = math.fabs(self.asymetry_moment / self.asymetry_moment_sqr_err)
        res_html = res_html + f'В нашем случае <font color="#000080">|As|/sAs = </font>'
        if asy_ratio < 3:
            res_html = res_html + f'<strong><font color="#006400">{asy_ratio:.02f}</font></strong> ' \
                                  f'<font color="#000080"><strong> меньше 3-х </strong></font>, ' \
                                  f'т.е. асимметрия формы распределения несущественная.'
        else:
            res_html = res_html + f'<font color="#ff0066"><strong>{asy_ratio:.02f}</font></strong> ' \
                                  f'<font color="#000080"><strong>больше 3-х</font></strong>, ' \
                                  f'т.е. <font color="#ff0066"><strong>асимметрия</strong></font> формы распределения' \
                                  f'<font color="#ff0066"><strong> существенная</strong></font>.'
        return res_html

    def calc_stat_0_build_all(self):
        self.calc_stat_1_values_list()
        self.calc_stat_2_distribution_center()
        self.calc_stat_3_variation()

    def calc_average(self):
        n = 0
        val = 0
        max_val = 0
        moda = 0
        vals_list = list(range(1, 100))
        for i in range(0, 99):
            vals_list[i] = 0

        row_data_by_minutes = []
        data_rows = len(self.rowdata)
        for data_i in range(0, data_rows - 1):
            if self.rowdata[data_i]['num_objects'] > 10:
                timedelta = self.rowdata[data_i]['dtime'] - self.rowdata[data_i - 1]['dtime']
                row_data_by_minutes.append(int(self.rowdata[data_i]['num_objects'] * 60 / timedelta.seconds))

        for event in row_data_by_minutes:
            if event > 10:
                n += 1
                val += event
                vals_list[event] = vals_list[event] + 1
            if event > max_val:
                max_val = event
        average = val / n
        # тут у нас есть  vals_list, номер элемента - хлебов в минуту = х, значение - количество раз = у
        # считаем средневзвешенную: sum( x * y ) / sum ( y )
        fi_sum = 0
        xfi_sum = 0
        max_val = 0
        median = 0
        for i in range(0, len(vals_list)):
            fi_sum = fi_sum + vals_list[i]
            xfi_sum = xfi_sum + i * vals_list[i]
            if vals_list[i] > max_val:
                max_val = vals_list[i]
                moda = i
        fi_sum_05 = fi_sum / 2
        if fi_sum != 0:
            average_weighted = xfi_sum / fi_sum
        else:
            average_weighted = 999999

        fi_sum_m = 0
        for i in range(1, len(vals_list)):
            fi_sum_m += vals_list[i]
            if median == 0 and fi_sum_m >= fi_sum_05:
                median = i

        print(vals_list)
        # self.print_histogram(vals_list)

        print('\n=== Показатели центра распределения. ===\n')
        print(f'Average: {average:.2f}, val={val}, n={n},  max={max_val}\n')
        print(f'\nСредняя взвешенная (выборочная средняя): {average_weighted:.2f}')
        print(f'\nМода: {moda}')
        print(f'Мода - наиболее часто встречающееся значение (хлебов в минуту): {moda} \n Максимальное значение '
              f'повторений при x = {moda} (f = {max_val}). Следовательно, мода равна {moda}')
        print(f'\nМедиана: {median} ')
        print(f'Медиана - значение хлебов в минуту, приходящееся на середину выборки: '
              f'в среднем {median} хлебов в минуту')
        print(f'Медиана служит хорошей характеристикой при ассиметричном распределении данных, т.к. даже при наличии '
              f'выбросов данных, медиана более устойчива к воздействию отклоняющихся данных.')
        print(f'В симметричных рядах распределения значение моды и медианы совпадают со средней величиной '
              f'(xср=Meдиана=Moда), а в умеренно асимметричных '
              f'они соотносятся таким образом: 3(Хср-Meдиана) ≈ Хср-Moда')
        print(f"То, что сравниваем с 3: {math.fabs((average_weighted - moda) / (average_weighted - median)):.2f}")
        if average_weighted == moda == median:
            print('В нашем случае, (xср=Meдиана=Moда) и следовательно, распределение симметрично. Отлично.')
        elif math.fabs((average_weighted - moda) / (average_weighted - median)) < 1:
            print('В нашем случае, соотношение Меньше 1'
                  f'{math.fabs((average_weighted - moda)  / (average_weighted - median))} '
                  f'следовательно, распределение является почти симметричным ')
        elif math.fabs((average_weighted - moda) / (average_weighted - median)) < 3:
            print('В нашем случае, соотношение: '
                  f'{math.fabs((average_weighted - moda)  / (average_weighted - median)):.2f} < 3, '
                  f'следовательно, распределение является умеренно асимметричным ')

        else:
            print(f'В нашем случае, соотношение: '
                  f'{math.fabs((average_weighted - moda)  / (average_weighted - median)):.2f} > 3, '
                  f'следовательно, распределение является асимметричным  :(')

        print('\n=== Показатели вариации. ===\n')
        print('\n-- Абсолютные показатели вариации. --\n')
        print('\nРазмах вариации = мах_х - мин_х  - вроде как нам он не нужен\n')

        fi_sum_d = 0
        x_i_minus_x_avg = 0
        for i in range(1, len(vals_list)):
            fi_sum_d += vals_list[i]
            x_i_minus_x_avg = x_i_minus_x_avg + (i - average_weighted) * (i - average_weighted) * vals_list[i]
        dispersia = x_i_minus_x_avg / fi_sum_d
        print(f'\nДисперсия.. = {dispersia:.2f} ')
        print('Дисперсия - характеризует меру разброса около ее среднего значения '
              '(мера рассеивания, т.е. отклонения от среднего). \n')
        dispersia_korrected = x_i_minus_x_avg / (fi_sum_d - 1)
        print(f'Несмещенная оценка дисперсии - состоятельная оценка дисперсии (S^2, исправленная дисперсия). '
              f'Дисперсия испр = {dispersia_korrected:.2f} ')
        sigma = math.sqrt(dispersia)
        print(f'\nСигма.. = {sigma:.2f} ')
        print(f'\n3 * Сигма.. = {3 * sigma:.2f},  -3 сигма - среднее + 3 сигма: {average_weighted - 3*sigma:.2f} '
              f'- {average_weighted:.2f} - {average_weighted + 3*sigma:.2f}')

    @staticmethod
    def print_histogram(in_list, is_html=None):
        """
        выводим гистограмму распределения в символьном виде (не график), чтобы глазами посмотреть на распределение
        во входном списке - распределение Х = индекс, У = значение in_list[Х]
        т.е. in_list[30] = 40 это значит, что 40 раз насчитали 30 хлебов в минуту
        """
        # высота гистограмы - 25 строк
        rows = 25
        # горизонтальный размер: 3 столбика на один элемент (чтобы подпись оси была норм.
        n = len(in_list)
        # левую границу определим автоматом ниже
        kol_min = -1
        max_val = 0

        res_html = '<p></p><p><strong><font color="#ff0066">=== Гистограмма распределения ===</strong></font></p>'
        res_html = res_html + '<p>'
        for i in range(0, n - 1):
            if in_list[i] > max_val:
                max_val = in_list[i]
        col_left_lvl = max_val / rows
        # определим левую границу гистограмы
        i = -1
        for x_i in in_list:
            i += 1
            if x_i > col_left_lvl and kol_min < 0:
                kol_min = i - 1
        # строим гисторгаму, результат - в строку по строкам
        for row in range(0, rows):
            row_str = '|'
            for col in range(kol_min, n):
                if in_list[col] / (max_val / rows) > (rows - row):
                    row_str = row_str + '###'
                else:
                    if row != rows - 1:
                        if is_html != 'html':
                            row_str = row_str + '   '
                        else:
                            row_str = row_str + '&nbsp;&nbsp;&nbsp;'
                    else:
                        row_str = row_str + '___'
            if is_html == 'html':
                res_html = res_html + f'<font color="#000080"><code>{row_str}</code></font><br>'
            else:
                print(row_str)
            # row_str = ''
        row_str = ''
        # подписываем ось X - значений
        for col in range(kol_min, n):
            row_str = row_str + f".{col:02d}"
        # собираем результат с осью
        if is_html == 'html':
            res_html = res_html + f'<font color="#000000"><code>|{row_str}</code></font><br>'
        else:
            print(row_str)
        return res_html


def main():
    camera_data = CameraData()  # инициализируем данные с камеры (загрузка данных из SQL через конструктор класса)

    # camera_data.report_working_time_build()  # в конструкторе запускаем формирование отчета
    print('\n')
    camera_data.report_working_time_print()
    print('\n')

    # camera_data.report_row_data_print()
    camera_data.report_working_time_ru_print()
    report_html = camera_data.report_working_time_html()  # собрали HTML для отчета на почту..
    # camera_data.calc_average()

    # вычисляем все показатели распределения (центры, , дисперсию, форму распределения)
    camera_data.calc_stat_0_build_all()

    # печатаем гистограмму в окне лога, без неё не инетресно
    camera_data.print_histogram(camera_data.vals_in_minute, 'print text')

    # собираем HTML для отчета в почту (гистограмму и параметры распределения хлебов в минуту)
    report_html = report_html + camera_data.print_histogram(camera_data.vals_in_minute, 'html')
    report_html = report_html + camera_data.report_distribution_stat_html()

    email = EmailHTML()
    email.get_message_html(report_html)
    # собственно, отправка электронного письма.
    email.send_email()


if __name__ == "__main__":
    main()