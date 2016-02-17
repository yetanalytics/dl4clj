(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/word2vec/Word2Vec.html"}
  dl4clj.models.word2vec.word2vec  
  (require [dl4clj.models.sequencevectors.sequence-vectors :as sequence-vectors])
  (:import [org.deeplearning4j.models.word2vec Word2Vec Word2Vec$Builder]))


(defn builder [{:keys [batch-size ;; This method defines mini-batch size (int)
                       epochs ;; This method defines number of epochs (iterations over whole training corpus) for training (int)
                       iterate ;; This method used to feed SequenceIterator, that contains training corpus, into ParagraphVectors (DocumentIterator OR SentenceIterator OR SequenceIterator<VocabWord>)
                       iterations ;; This method defines number of iterations done for each mini-batch during training (int)
                       layer-size ;; This method defines number of dimensions for output vectors (int)
                       learning-rate ;; This method defines initial learning rate for model training (double)
                       lookup-table ;; This method allows to define external WeightLookupTable to be used (WeightLookupTable<VocabWord> lookupTable)
                       min-learning-rate ;; This method defines minimal learning rate value for training (double)
                       min-word-frequency ;; This method defines minimal word frequency in training corpus. (int)
                       model-utils ;; Sets ModelUtils that gonna be used as provider for utility methods: similarity(), wordsNearest(), accuracy(), etc (ModelUtils<VocabWord> modelUtils)
                       negative-sample ;; This method defines whether negative sampling should be used or not (double)
                       reset-model ;; This method defines whether model should be totally wiped out prior building, or not (boolean)
                       sampling ;; This method defines whether subsampling should be used or not (double)
                       seed ;; This method defines random seed for random numbers generator (long)
                       stop-words ;; This method defines stop words that should be ignored during training (java.util.Collection<VocabWord>OR java.util.List<java.lang.String>)
                       tokenizer-factory ;; This method defines TokenizerFactory to be used for strings tokenization during training PLEASE NOTE: If external VocabCache is used, the same TokenizerFactory should be used to keep derived tokens equal. (TokenizerFactory tokenizerFactory)
                       train-elements-representation ;; This method is hardcoded to TRUE, since that's whole point of Word2Vec (boolean)
                       train-sequences-representation ;; This method is hardcoded to FALSE, since that's whole point of Word2Vec (boolean)
                       use-ada-grad ;; This method defines whether adaptive gradients should be used or not (boolean)
                       vocab-cache ;; This method allows to define external VocabCache to be used (VocabCache<VocabWord> vocabCache)
                       window-size ;; This method defines context window size (int)
                       workers ;; This method defines maximum number of concurrent threads available for training (int)
                       ]
                :or {}
                :as opts}] 
  (let [b ^Word2Vec$Builder (sequence-vectors/builder (Word2Vec$Builder.) (dissoc opts
                                                                                  :batch-size
                                                                                  :epochs
                                                                                  :iterate
                                                                                  :iterations
                                                                                  :layer-size
                                                                                  :learning-rate
                                                                                  :lookup-table
                                                                                  :min-learning-rate
                                                                                  :min-word-frequency
                                                                                  :model-utils
                                                                                  :negative-sample
                                                                                  :reset-model
                                                                                  :sampling
                                                                                  :seed
                                                                                  :stop-words
                                                                                  :tokenizer-factory
                                                                                  :train-elements-representation
                                                                                  :train-sequences-representation
                                                                                  :use-ada-grad
                                                                                  :vocab-cache
                                                                                  :window-size
                                                                                  :workers))]
    (when (or batch-size (contains? opts :batch-size))
      (.batchSize b (int batch-size)))
    (when (or epochs (contains? opts :epochs))
      (.epochs b (int epochs)))
    (when (or iterate (contains? opts :iterate))
      (.iterate b iterate))
    (when (or iterations (contains? opts :iterations))
      (.iterations b iterations))
    (when (or layer-size (contains? opts :layer-size))
      (.layerSize b layer-size))
    (when (or learning-rate (contains? opts :learning-rate))
      (.learningRate b learning-rate))
    (when (or lookup-table (contains? opts :lookup-table))
      (.lookupTable b lookup-table))
    (when (or min-learning-rate (contains? opts :min-learning-rate))
      (.minLearningRate b (double min-learning-rate)))
    (when (or min-word-frequency (contains? opts :min-word-frequency))
      (.minWordFrequency b min-word-frequency))
    (when (or model-utils (contains? opts :model-utils))
      (.modelUtils b model-utils))
    (when (or negative-sample (contains? opts :negative-sample))
      (.negativeSample b (double negative-sample)))
    (when (or reset-model (contains? opts :reset-model))
      (.resetModel b reset-model))
    (when (or sampling (contains? opts :sampling))
      (.sampling b sampling))
    (when (or seed (contains? opts :seed))
      (.seed b seed))
    (when (or stop-words (contains? opts :stop-words))
      (.stopWords b stop-words))
    (when (or tokenizer-factory (contains? opts :tokenizer-factory))
      (.tokenizerFactory b tokenizer-factory))
    (when (or train-elements-representation (contains? opts :train-elements-representation)) 
      (.trainElementsRepresentation b train-elements-representation))
    (when (or train-sequences-representation (contains? opts :train-sequences-representation))
      (.trainSequencesRepresentation b train-sequences-representation))
    (when (or use-ada-grad (contains? opts use-ada-grad))
      (.useAdaGrad b use-ada-grad))
    (when (or vocab-cache (contains? opts vocab-cache))
      (.vocabCache b vocab-cache))
    (when (or window-size (contains? opts window-size))
      (.windowSize b window-size))
    (when (or workers (contains? opts workers))
      (.workers b workers))
    b
    ))


(defn word2vec [opts]
  (.build ^Word2Vec$Builder (builder opts)))



