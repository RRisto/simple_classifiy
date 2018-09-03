import pickle

from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import pyLDAvis.gensim


class CustomLda(object):
    def __init__(self, data=None, dictionary=None):
        """ initialize, data should be provided, only when unpickling class object it is not needed!"""
        self.data = data
        self.model = None
        self.num_topics = None
        self.iterations = None
        self.random_state = None
        self.dictionary = dictionary
        if self.data is not None:
            if self.dictionary is None:
                self.dictionary = Dictionary(self.data)
            self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        else:
            self.dictionary = None
            self.corpus = None
        self.distributed = None
        self.chuncksize = None
        self.passes = None
        self.update_every = None
        self.alpha = None
        self.eta = None
        self.decay = None
        self.offset = None
        self.eval_every = None
        self.gamma_threshold = None
        self.minimum_probability = None
        self.ns_conf = None
        self.minimum_phi_value = None
        self.per_word_topics = None
        self.num_topics = None
        self.iterations = None
        self.random_state = None
        self.model = None
        self.coherence_model = None
        self.coherence = None
        self.coherence_type = None

    def train(self, num_topics, iterations=1500, random_state=1,
              distributed=False, chunksize=2000, passes=1, update_every=1, alpha='symmetric',
              eta=None, decay=0.5, offset=1.0, eval_every=10, gamma_threshold=0.001,
              minimum_probability=0.01, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False,
              workers=1):
        """train lda model. If workers >1, goes multicore"""

        self.distributed = distributed
        self.chuncksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.eval_every = eval_every
        self.gamma_threshold = gamma_threshold
        self.minimum_probability = minimum_probability
        self.ns_conf = ns_conf
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.num_topics = num_topics
        self.iterations = iterations
        self.random_state = random_state
        self.workers = workers

        if self.workers > 1:
            self.model = LdaMulticore(workers=3, corpus=self.corpus, id2word=self.dictionary,
                                      iterations=self.iterations, num_topics=self.num_topics,
                                      random_state=self.random_state,  # distributed=self.distributed,
                                      chunksize=self.chuncksize,
                                      passes=self.passes,  # update_every= self.update_every,
                                      alpha=self.alpha, eta=self.eta, decay=self.decay,
                                      offset=self.offset, eval_every=self.eval_every,
                                      gamma_threshold=self.gamma_threshold,
                                      minimum_probability=self.minimum_probability,  # ns_conf=self.ns_conf,
                                      minimum_phi_value=self.minimum_phi_value, per_word_topics=self.per_word_topics)
        else:
            self.model = LdaModel(corpus=self.corpus, id2word=self.dictionary,
                                  iterations=self.iterations, num_topics=self.num_topics,
                                  random_state=self.random_state, distributed=self.distributed,
                                  chunksize=self.chuncksize,
                                  passes=self.passes, update_every=self.update_every,
                                  alpha=self.alpha, eta=self.eta, decay=self.decay,
                                  offset=self.offset, eval_every=self.eval_every, gamma_threshold=self.gamma_threshold,
                                  minimum_probability=self.minimum_probability, ns_conf=self.ns_conf,
                                  minimum_phi_value=self.minimum_phi_value, per_word_topics=self.per_word_topics)
        print('Trained!')

    def _train_coherence_model(self, coherence_type='u_mass'):
        """could be made on top of model to get coherence, type could be 'u_mass' or 'c_v'"""
        self.coherence_model = CoherenceModel(model=self.model, texts=self.data, dictionary=self.dictionary,
                                              coherence=coherence_type)

    def _calculate_coherence(self, coherence_type='u_mass'):
        self._train_coherence_model(coherence_type=coherence_type)
        self.coherence = self.coherence_model.get_coherence()

    def get_coherence(self, coherence_type='u_mass'):
        if coherence_type != self.coherence_type:
            self._calculate_coherence(coherence_type=coherence_type)
        return self.coherence

    def get_topic_terms(self, num, topn=10):
        return self.model.get_topic_terms(num, topn=topn)

    def get_preplexity(self):
        return self.model.log_perplexity(self.corpus)

    def get_topics(self, num):
        return self.model.show_topics(num)

    def _make_visualization(self):
        """prepare visualisation for display/saving"""
        return pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary, sort_topics=False)

    def display(self):
        """display LDAvis in notebook"""
        visualisation = self._make_visualization()
        return pyLDAvis.display(visualisation)

    def save_ldavis(self, filename='topic.html'):
        """save LDAvis to .html"""
        ldavis = self._make_visualization()
        pyLDAvis.save_html(ldavis, filename)

    def save_lda(self, filename):
        """save lda model only"""
        self.model.save(filename)

    def pickle(self, filename):
        """save class instance to file"""
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def unpickle(filename):
        """read class instance from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def predict_topic(self, doc_list):
        """predict topic of document list (consists of strings"""
        topic_list = []
        for doc in doc_list:
            bow = self.dictionary.doc2bow(str(doc).split())
            topics_probs = self.model.get_document_topics(bow)
            topics_probs.sort(key=lambda tup: tup[1], reverse=True)
            topic_list.append(topics_probs)
        return topic_list
