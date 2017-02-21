(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/models/sequencevectors/SequenceVectors.html"}
  dl4clj.models.sequencevectors.sequence-vectors
  (:import [org.deeplearning4j.models.sequencevectors SequenceVectors SequenceVectors$Builder]
           [org.deeplearning4j.models.sequencevectors.sequence Sequence]))

(defn builder
  ([]
   (SequenceVectors.))
  ([opts]
   (builder (SequenceVectors$Builder.) opts))
  ([^SequenceVectors$Builder b {:keys [batch-size ;; This method defines batchSize option, viable only if iterations > 1 (int)
                                       elements-learning-algorithm ;; * Sets specific LearningAlgorithm as Elements Learning Algorithm (String or ElementsLearningAlgorithm)
                                       epochs ;; This method defines how much iterations should be done over whole training corpus during modelling (int)
                                       iterate ;; This method defines SequenceIterator to be used for model building (SequenceIterator<T>)
                                       iterations ;; This method defines how much iterations should be done over batched sequences. (int)
                                       layer-size ;; This method defines number of dimensions for outcome vectors. (int)
                                       learning-rate ;; This method defines initial learning rate. (double)
                                       lookup-table ;; You can pass externally built WeightLookupTable, containing model weights and vocabulary. (WeightLookupTable<T> lookupTable)
                                       min-learning-rate ;; This method defines minimum learning rate after decay being applied. (double)
                                       min-word-frequency ;; This method defines minimal element frequency for elements found in the training corpus. (int)
                                       model-utils ;; ModelUtils implementation, that will be used to access model. (ModelUtils<T>)
                                       negative-sample ;; This method defines negative sampling value for skip-gram algorithm. (double)
                                       preset-tables ;; This method creates new WeightLookupTable and VocabCache if there were none set ()
                                       reset-model ;; This method defines, should all model be reset before training. (boolean)
                                       sampling ;; This method defines sub-sampling threshold. (double)
                                       seed     ;; Sets seed for random numbers generator. (long)
                                       sequence-learning-algorithm ;; Sets specific LearningAlgorithm as Sequence Learning Algorithm (SequenceLearningAlgorithm<T> or STring)
                                       stop-words ;; You can provide collection of objects to be ignored, and excluded out of model Please note Object labels and hashCode will be used for filtering (java.util.Collection<T> OR java.util.List<java.lang.String> stopList)
                                       train-elements-representation                   ;; (boolean)
                                       train-sequences-representation                  ;; (boolean)
                                       use-ada-grad ;; This method defines if Adaptive Gradients should be used in calculations (boolean)
                                       vocab-cache ;; You can pass externally built vocabCache object, containing vocabulary (VocabCache<T>)
                                       window-size ;; Sets window size for skip-Gram training (int)
                                       workers ;; Sets number of worker threads to be used in calculations (int)
                                       ]
                                :or {}
                                :as opts}]
   (when (or batch-size (contains? opts batch-size))
     (.batchSize b batch-size))
   (when (or elements-learning-algorithm (contains? opts elements-learning-algorithm))
     (.elementsLearningAlgorithm b elements-learning-algorithm))
   (when (or epochs (contains? opts epochs))
     (.epochs b epochs))
   (when (or iterate (contains? opts iterate))
     (.iterate b iterate))
   (when (or iterations (contains? opts iterations))
     (.iterations b iterations))
   (when (or layer-size (contains? opts layer-size))
     (.layerSize b layer-size))
   (when (or learning-rate (contains? opts learning-rate))
     (.learningRate b learning-rate))
   (when (or lookup-table (contains? opts lookup-table))
     (.lookupTable b lookup-table))
   (when (or min-learning-rate (contains? opts min-learning-rate))
     (.minLearningRate b min-learning-rate))
   (when (or min-word-frequency (contains? opts min-word-frequency))
     (.minWordFrequency b min-word-frequency))
   (when (or model-utils (contains? opts model-utils))
     (.modelUtils b model-utils))
   (when (or negative-sample (contains? opts negative-sample))
     (.negativeSample b negative-sample))
   (when (or preset-tables (contains? opts preset-tables))
     (.presetTables b preset-tables))
   (when (or reset-model (contains? opts reset-model))
     (.resetModel b reset-model))
   (when (or sampling   (contains? opts sampling  ))
     (.sampling   b sampling  ))
   (when (or seed       (contains? opts seed      ))
     (.seed       b seed      ))
   (when (or sequence-learning-algorithm (contains? opts sequence-learning-algorithm))
     (.sequenceLearningAlgorithm b sequence-learning-algorithm))
   (when (or stop-words (contains? opts stop-words))
     (.stopWords b stop-words))
   (when (or train-elements-representation (contains? opts train-elements-representation))
     (.trainElementsRepresentation b train-elements-representation))
   (when (or train-sequences-representation (contains? opts train-sequences-representation))
     (.trainSequencesRepresentation b train-sequences-representation))
   (when (or use-ada-grad (contains? opts use-ada-grad))
     (.useAdaGrad b use-ada-grad))
   (when (or vocab-cache (contains? opts vocab-cache))
     (.vocabCache b vocab-cache))
   (when (or window-size (contains? opts window-size))
     (.windowSize b window-size))
   (when (or workers (contains? opts workers   ))
     (.workers  b workers   ))
   b))


(defn build-vocab
  "Builds vocabulary from provided SequenceIterator instance"
  [^SequenceVectors this]
  (.buildVocab this)
  this)

(defn fit
  [^SequenceVectors this]
  (.fit this)
  this)

;; (defn train-sequence [^SequenceVectors this ^Sequence sequence ^java.util.concurrent.atomic.AtomicLong next-random ^Double alpha]
;;   (.trainSequence this sequence next-random alpha))
