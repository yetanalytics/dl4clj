(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/paragraphvectors/ParagraphVectors.html"}
  dl4clj.models.paragraphvectors.paragraph-vectors
  (:require [dl4clj.models.word2vec.word2vec :as word2vec])
  (:import [org.deeplearning4j.models.paragraphvectors ParagraphVectors ParagraphVectors$Builder]))


(defn builder [{:keys [batch-size ;; "This method defines mini-batch size" (int)
                       epochs ;; "This method defines number of epochs (iterations over whole training corpus) for training" (int)
                       index  ;; (InvertedIndex<VocabWord>)
                       iterate ;; "This method used to feed DocumentIterator, that contains training corpus, into ParagraphVectors" (DocumentIterator OR LabelAwareDocumentIterator OR LabelAwareIterator OR LabelAwareSentenceIterator OR SentenceIterator OR SequenceIterator<VocabWord> OR iterator)
                       iterations ;; "This method defines number of iterations done for each mini-batch during training" (int)
                       labels-source ;; "This method attaches pre-defined labels source to ParagraphVectors" (LabelsSource)
                       layer-size ;; "This method defines number of dimensions for output vectors" (int)
                       learning-rate ;; "This method defines initial learning rate for model training" (double)
                       lookup-table ;; "This method allows to define external WeightLookupTable to be used" (WeightLookupTable<VocabWord>)
                       min-learning-rate ;; "This method defines minimal learning rate value for training " (double)
                       min-word-frequency ;; " This method defines minimal word frequency in training corpus. (int)
                       model-utils ;; "Sets ModelUtils that gonna be used as provider for utility methods: similarity(), wordsNearest(), accuracy(), etc" (ModelUtils<VocabWord>)
                       negative-sample ;; "This method defines whether negative sampling should be used or not" (double)
                       reset-model ;; "This method defines whether model should be totally wiped out prior building, or not " (boolean)
                       sampling ;; "This method defines whether subsampling should be used or not" (double)
                       seed ;; "This method defines random seed for random numbers generator" (long)
                       stop-words ;; "This method defines stop words that should be ignored during training" (java.util.Collection<VocabWord> OR java.util.List<java.lang.String>)
                       tokenizer-factory ;; "This method defines TokenizerFactory to be used for strings tokenization during training PLEASE NOTE: If external VocabCache is used, the same TokenizerFactory should be used to keep derived tokens equal." (TokenizerFactory)
                       train-elements-representation ;; "This method defines, if words representation should be build together with documents representations." (boolean)
                       train-sequences-representation ;; This method is hardcoded to TRUE, since that's whole point of ParagraphVectors (boolean)
                       train-word-vectors ;; "This method defines, if words representations should be build together with documents representations." (boolean)
                       use-ada-grad ;; "This method defines whether adaptive gradients should be used or not" (boolean)
                       vocab-cache ;; "This method allows to define external VocabCache to be used" (VocabCache<VocabWord>)
                       window-size ;; "This method defines context window size" (int)
                       workers ;; method defines maximum number of concurrent threads available for training (int)
                       ]
                :or {train-sequences-representation true}
                :as opts}]
  (let [b ^ParagraphVectors$Builder (word2vec/builder
                                     (ParagraphVectors$Builder.)
                                     (dissoc opts
                                             :batch-size
                                             :epochs
                                             :index
                                             :iterate
                                             :iterations
                                             :labels-source
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
                                             :train-word-vectors
                                             :use-ada-grad
                                             :vocab-cache
                                             :window-size
                                             :workers))]
    (when (or batch-size (contains? opts :batch-size))
      (.batchSize b (int batch-size)))
    (when (or epochs (contains? opts :epochs))
      (.epochs b (int epochs)))
    (when (or index (contains? opts :index))
      (.index b index))
    (when (or iterate (contains? opts :iterate))
      (.iterate b iterate))
    (when (or iterations (contains? opts :iterations))
      (.iterations b (int iterations)))
    (when (or labels-source (contains? opts :labels-source))
      (.labelsSource b labels-source))
    (when (or layer-size (contains? opts :layer-size))
      (.layerSize b (int layer-size)))
    (when (or learning-rate (contains? opts :learning-rate))
      (.learningRate b (double learning-rate)))
    (when (or lookup-table (contains? opts :lookup-table))
      (.lookupTable b lookup-table))
    (when (or min-learning-rate (contains? opts :min-learning-rate))
      (.minLearningRate b (double min-learning-rate)))
    (when (or min-word-frequency (contains? opts :min-word-frequency))
      (.minWordFrequency b (int min-word-frequency)))
    (when (or model-utils (contains? opts :model-utils))
      (.modelUtils b model-utils))
    (when (or negative-sample (contains? opts :negative-sample))
      (.negativeSample b (double negative-sample)))
    (when (or reset-model (contains? opts :reset-model))
      (.resetModel b (boolean reset-model)))
    (when (or sampling (contains? opts :sampling))
      (.sampling b (double sampling)))
    (when (or seed (contains? opts :seed))
      (.seed b (long seed)))
    (when (or stop-words (contains? opts :stop-words))
      (.stopWords b stop-words))
    (when (or tokenizer-factory (contains? opts :tokenizer-factory))
      (.tokenizerFactory b tokenizer-factory))
    (when (or train-elements-representation (contains? opts :train-elements-representation))
      (.trainElementsRepresentation b (boolean train-elements-representation)))
    (when (or train-sequences-representation (contains? opts :train-sequences-representation))
      (.trainSequencesRepresentation b (boolean train-sequences-representation)))
    (when (or train-word-vectors (contains? opts :train-word-vectors))
      (.trainWordVectors b (boolean train-word-vectors)))
    (when (or use-ada-grad (contains? opts :use-ada-grad))
      (.useAdaGrad b (boolean use-ada-grad)))
    (when (or vocab-cache (contains? opts :vocabCache))
      (.vocabCache b vocab-cache))
    (when (or window-size (contains? opts :window-size))
      (.windowSize b (int window-size)))
    (when (or workers (contains? opts :workers))
      (.workers b (int workers)))
    b))

(defn paragraph-vectors [opts]
  (.build ^ParagraphVectors$Builder (builder opts)))
