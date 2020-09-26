import tensorflow as tf
import numpy as np
import os
import re
import email
from bs4 import BeautifulSoup
import joblib
import pathlib
import pickle
from itertools import cycle


class Scam_parser():

    def __init__(self, email_token, url_token, money_token,
                 tel_token, name_token, relative_token,
                 start_token, end_token, unknown_token):

        self.stored_tokens = None
        self.seperator = '\n\n\n\n' + '%' * 50 + '\n\n\n\n'

        token_list = [email_token, url_token, money_token,
                      tel_token, name_token, relative_token,
                      start_token, end_token, unknown_token]

        assert len(set(token_list)) == len(token_list)

        self.email_token = email_token
        self.url_token = url_token
        self.money_token = money_token
        self.tel_token = tel_token
        self.name_token = name_token
        self.relative_token = relative_token
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

        self.money_words = [
            'one', 'two', 'three', 'four', 'five',
            'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'tweleve', 'thirteen', 'fourteen', 'fifteen',
            'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
            'thirty', 'fou?rty', 'fifty', 'sixty', 'seventy',
            'eighty', 'ninety', 'hundreds?', 'thousands?', 'millions?',
            'billions?', 'united', 'states?', 'dollars?', 'only',
            'point', 'pounds?', 'British', 'Sterling', 'usa?d?',
            'and', 'American'
        ]

        self.name_valedictions = [
            r'Your\'?s\ +in\ +Christ',
            r'Sincerely',
            r'Regards\ +and\ +respect',
            r'regards?',
            r'Your\'?s\ +Truly',
            r'Your\'?s\ +Faithfully',
            r'Best\ +Wishes',
            r'Thanks\ +in\ +advance',
            r'yours',
        ]

        self.name_pronouns = [
            r'sister', r'auditor', r'barrister', r'barr?',
            r'CAPT', r'ENGINEER', r'Engr?', r'DR',
            r'lt', r'general', r'gen', r'Hon',
            r'Madam', r'MR?S', r'MR', r'Miss',
            r'Professor', r'Prof', r'Revd', r'Rev',
            r'DEACON', r'retired', r'ret', r'Pastor',
            r'Princess', r'Prince', r'Senator', r'Sen',
            r'Sir', r'Major', r'Maj', r'Col',
            r'Chief', r'Evangelist', r'late', r'president',
            r'minister', r'fr', r'lady', r'husband',
            r'son', r'leader', r'Brigadier'
        ]

        self.tel_regex = re.compile(
            r'(?:\+\s*)?(?:\(\s*\d+\s*\)(?:\s*\-?\s*\d+)+|\d+(?:(?:\s+|\s*\-\s*)\d+)+|\d{7,})')
        self.apost_regex = re.compile(
            "(\'s|\'re|\'ve|\'m|\'t|\'ll|\'d)", flags=re.IGNORECASE)
        self.line_regex = re.compile(
            r'(?:\.{4,}|\,{4,}|\={4,}|\-{4,}|\_{4,}|\*{4,})')
        self.first_line_regex = re.compile(r'From\ r\ \ ')
        self.email_regex = re.compile(r'[\w\.-]+@[\w\.-]+(?:\.[\w]+)+')
        self.url_regex = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

        self.money_words_loop = '(?:(?:' + '|'.join(self.money_words) + \
            ')' + '[\s\,\.\-]*' + '){2,}'
        self.parenth_regex = '[\(\[\{][^\)\]\}]{2,60}[\)\]\}]'

        self.money_regex1 = re.compile(
            '(?:' + self.parenth_regex + ')?[\s\,\.]*' + self.money_words_loop + '[\s\,\.]*(?:' + self.parenth_regex + ')?', flags=re.IGNORECASE)

        self.money_regex2 = re.compile(
            '(?:(?:usd?|\$|\£|\€|¥|GBP?)\s*)+\d+(?:[.,]\d+)*\s*(?:(?:millions?|billions?|m|u\.?s\.?d?|dollars?|only|british|pounds?|sterling|yuan|\.|\,)\s*)*(?:[\s\,\.]*[\(\[\{][\w\s\,\.\-\&]+[\)\]\}])?', flags=re.IGNORECASE)

        self.name_regex_start = '(?:' + '|'.join(self.name_valedictions) + ')'
        self.name_regex = self.name_regex_start + \
            '[\ \.\,]*(?:\n[\ \.\,]*)+(\w[\w\ \&\.\,\-\:\/]+)'
        self.pronoun_regex_loop = '(?:' + '|'.join(self.name_pronouns) + ')'
        self.pronoun_regex = '(?:' + self.pronoun_regex_loop + '[\.\,\s\/]+)*'
        self.bottom_pronoun_regex = '^[\.\,\s]*(?:' + \
            self.pronoun_regex_loop + '[\.\,\s\/]+)*'

        self.wrong_relative_regex = re.compile(
            r'I\s+am\s+' + self.relative_token.replace('^', '\^'), flags=re.IGNORECASE)

    def fix_punct(self, email):

        symbols = r':;!?%&\(\)\[\]\/'

        email = re.sub(r'(['+symbols+'])', r' \1 ', email)
        email = re.sub(r'\t', ' ', email)
        email = re.sub(r'\n+|\r', ' \n ', email)
        email = self.line_regex.sub('', email)

        email = re.sub(r'(\D)([\.\,])(\D)', r'\1 \2 \3', email)
        email = re.sub(r'(\d)([\.\,])(\D)', r'\1 \2 \3', email)
        email = re.sub(r'(\D)([\.\,])(\d)', r'\1 \2 \3', email)
        email = re.sub('[‘’]', "'", email)
        email = self.apost_regex.sub(r' \1 ', email)

        email = self.start_token + ' ' + email + ' ' + self.end_token

        return email

    def mask_tokens(self, email):

        def mask_emails(email):

            local_emails = []

            def replace(match):

                local_emails.append(match.group(0))
                return ' ' + self.email_token + ' '

            email = self.email_regex.sub(replace, email)

            return email, set(local_emails)

        def mask_urls(email):

            local_urls = []

            def replace(match):

                local_urls.append(match.group(0))
                return ' ' + self.url_token + ' '

            email = self.url_regex.sub(replace, email)

            return email, set(local_urls)

        def mask_money(email):

            local_money = []

            def valid_money(string):

                return True if re.search(r'[\(\[\{]', string) and re.search(r'[\)\]\}]', string) else False

            def replace1(match):

                if valid_money(match.group(0)):
                    local_money.append(match.group(0))
                    return ' ' + self.money_token + ' '
                else:
                    return match.group(0)

            def replace2(match):

                local_money.append(match.group(0))
                return ' ' + self.money_token + ' '

            email = self.money_regex1.sub(replace1, email)

            email = self.money_regex2.sub(replace2, email)

            return email, set(local_money)

        def mask_tels(email):

            def valid_tel(tel):

                return len(re.findall(r'\d', tel)) >= 9

            local_tels = []

            def replace(match):

                if valid_tel(match.group(0)):
                    local_tels.append(match.group(0))
                    return ' ' + self.tel_token + ' '
                else:
                    return match.group(0)

            email = self.tel_regex.sub(replace, email)
            return email, set(local_tels)

        def mask_names(email):

            local_relatives = []

            def replace_relative(match):

                local_relatives.append(match.group(0))
                return ' ' + self.relative_token + ' '

            local_name = re.findall(
                self.name_regex, email, flags=re.IGNORECASE)

            if not local_name:
                return email, None

            local_name = local_name[-1]

            # remove pronouns
            local_name = re.sub(self.bottom_pronoun_regex,
                                '', local_name, flags=re.IGNORECASE)

            split_local_name = re.findall(r'\w+', local_name)
            if len(split_local_name) <= 1:
                return email, None

            surname = split_local_name[-1]

            local_name_regex = self.pronoun_regex + \
                '[\.\,\s\-]*'.join(re.findall('\w', local_name))

            email = re.sub(local_name_regex, ' ' +
                           self.name_token + ' ', email, flags=re.IGNORECASE)

            if len(surname) > 2:

                local_relative_regex = self.pronoun_regex + \
                    '(?:\w+[\.\,\s\-]*){0,2}[\.\,\s\-]+' + surname
                email = re.sub(local_relative_regex,
                               replace_relative, email, flags=re.IGNORECASE)
                email = self.wrong_relative_regex.sub(
                    'I am  ' + self.name_token, email)

            return email, (local_name, local_relatives)

        email, local_emails = mask_emails(email)
        email, local_urls = mask_urls(email)
        email, local_money = mask_money(email)
        email, local_tels = mask_tels(email)
        email, local_names = mask_names(email)

        local_tokens = {}
        local_tokens['email'] = local_emails
        local_tokens['url'] = local_urls
        local_tokens['money'] = local_money
        local_tokens['tel'] = local_tels
        local_tokens['name'] = local_names

        return email, local_tokens

    def split_emails(self, string):

        email_list = self.first_line_regex.split(string)[1:]
        email_list = list(map(lambda x: 'From r  ' + x, email_list))

        return email_list

    def get_email_body(self, string):

        body = email.message_from_string(string).get_payload()

        return body

    def replace_hex(self, string):

        def hex2ascii(match):
            return chr(int('0x' + match.group()[1:], 16))

        ascii_ = re.sub(r'\=[0-9A-F][0-9A-F]', hex2ascii, string)

        return ascii_

    def remove_newline(self, string):

        string = re.sub(r'\=\n', '', string)

        return string

    def remove_html(self, string):

        string = BeautifulSoup(string).get_text('\n')

        return string

    def preprocess_email(self, email):

        # remove metadata
        email = self.get_email_body(email)
        # replace hex characters with ascii
        email = self.replace_hex(email)
        # remove noise newlines
        email = self.remove_newline(email)
        # transform html to text
        email = self.remove_html(email)
        # mask certain tokens
        email, local_tokens = self.mask_tokens(email)

        email = self.fix_punct(email)

        return email, local_tokens

    def prepreprocess_corpus(self, src_filename, dst_filename=None):

        corpus = open(src_filename, 'r', encoding='ISO-8859-1',
                      newline="\n").read()
        assert len(corpus) > 1

        email_list = self.split_emails(corpus)
        preprocessed = joblib.Parallel(n_jobs=joblib.cpu_count())(
            joblib.delayed(self.preprocess_email)(e) for e in email_list)

        email_list, tokens_list = zip(*preprocessed)

        stored_tokens = {}
        for key in tokens_list[0].keys():
            if key != 'name':
                stored_tokens[key] = list(
                    set.union(*[d[key] for d in tokens_list]))
            else:
                name_list = [d[key] for d in tokens_list if d[key] != None]
                max_len = max([len(x[1]) for x in name_list])
                stored_tokens[key] = [[] for _ in range(max_len + 1)]
                for tuple_ in name_list:
                    n_rel = len(tuple_[1])
                    stored_tokens[key][n_rel].append(tuple_)
                for list_ in stored_tokens[key]:
                    assert len(list_) > 0
            print(
                f'Detected {len(stored_tokens[key])} different instances of "{key}".')

        self.stored_tokens = stored_tokens

        preprocessed_corpus = self.seperator.join(email_list)

        if not dst_filename is None:
            with open(dst_filename, 'w', encoding='ISO-8859-1') as file:
                file.truncate()
                file.write(preprocessed_corpus)

        return email_list, stored_tokens

    def save_features(self, features, filename):

        np.save(filename, features)

    def load_features(self, filename):

        return np.load(filename)

    def preprocess_dataset(self, corpus_path, n_words, npy_dir, tokenizer=None):

        def empty_filter(x):

            return len(x) > 2

        assert pathlib.Path(corpus_path).is_file()
        assert pathlib.Path(npy_dir).is_dir()

        preprocessed_emails, stored_tokens = self.prepreprocess_corpus(
            corpus_path)

        if tokenizer is None:

            tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=n_words - 1,
                filters='"“”‘’#*+-–=<>_`{|}~\n',
                oov_token=self.unknown_token
            )

            tokenizer.fit_on_texts(preprocessed_emails)

        elif isinstance(tokenizer, str):
            assert pathlib.Path(tokenizer).is_file()
            with open(tokenizer, 'rb') as handle:
                tokenizer = pickle.load(handle)
            assert isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer)
        else:
            assert isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer)

        feature_list = tokenizer.texts_to_sequences(preprocessed_emails)
        feature_list = list(filter(empty_filter, feature_list))
        feature_list = list(map(lambda x: np.array(x), feature_list))

        npy_filenames = [os.path.join(npy_dir, str(f) + '.npy')
                         for f in list(range(len(feature_list)))]

        for features, filename in zip(feature_list, npy_filenames):

            self.save_features(features, filename)

        tokenizer_path = os.path.join(npy_dir, 'tokenizer.pickle')
        stored_tokens_path = os.path.join(npy_dir, 'stored_tokens.pickle')
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(stored_tokens_path, 'wb') as handle:
            pickle.dump(stored_tokens, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        print('Finished preprocessing dataset')
        print(f'The npy files are stored at {npy_dir}')
        print(f'The tokenizer is stored at {tokenizer_path}')
        print(f'The tokens dictionary is stored at {stored_tokens_path}')

    def get_tf_dataset(self, file_directory, batch_size, n_samples=None):

        filenames = list(pathlib.Path(file_directory).rglob('*.npy'))
        assert len(filenames) > 0

        if not n_samples is None:
            assert isinstance(n_samples, int)
            n_samples = min(n_samples, len(filenames))
            filenames = np.random.choice(
                filenames, n_samples, replace=False).tolist()

        buffer_size = len(filenames)
        assert buffer_size > 0

        feature_list = list(map(self.load_features, filenames))
        features_ragged = tf.ragged.constant(feature_list)

        tf_dataset = tf.data.Dataset.from_tensor_slices((features_ragged))
        tf_dataset = tf_dataset.cache()
        tf_dataset = tf_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True)
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return tf_dataset

    def fill_masks(self, text, stored_tokens):

        text = re.sub(self.start_token, '', text)
        text = re.sub(self.end_token, '', text)

        money = np.random.choice(stored_tokens['money']).lower()
        url = np.random.choice(stored_tokens['url'])
        email = np.random.choice(stored_tokens['email'])
        tel = np.random.choice(stored_tokens['tel'])

        text = re.sub(self.money_token, money, text)
        text = re.sub(self.url_token, url, text)
        text = re.sub(self.email_token, email, text)
        text = re.sub(self.tel_token, tel, text)

        n_rel = len(re.findall(self.relative_token, text))
        n_rel_idx = min(n_rel, len(stored_tokens['name']) - 1)
        choice_idx = np.random.randint(
            0, len(stored_tokens['name'][n_rel_idx]))
        name_tuple = stored_tokens['name'][n_rel_idx][choice_idx]

        name = name_tuple[0].lower()
        rel_list = [x.lower() for x in name_tuple[1]]

        text = re.sub(self.name_token, name, text)
        text = re.sub(self.relative_token, lambda m,
                      i=cycle(rel_list): next(i), text)

        return text

    def features_to_text(self, features, tokenizer, stored_tokens=None):

        # features -> (batch_size, max_len)

        def remove_padding(t): return t.partition(self.end_token)[0]
        def add_nl(t): return re.sub(r'([\.\!\;])', r'\1\n', t)
        def fill_masks(t): return self.fill_masks(t, stored_tokens)

        text_list = tokenizer.sequences_to_texts(features)
        text_list = list(map(remove_padding, text_list))
        text_list = list(map(add_nl, text_list))

        if not stored_tokens is None:
            assert isinstance(stored_tokens, dict)
            text_list = list(map(fill_masks, text_list))

        return text_list

    @staticmethod
    def build_from_config(config):

        parser = Scam_parser(email_token=config.email_token, url_token=config.url_token,
                             money_token=config.money_token, tel_token=config.tel_token,
                             name_token=config.name_token, relative_token=config.relative_token,
                             start_token=config.start_token, end_token=config.end_token,
                             unknown_token=config.unknown_token)

        return parser
