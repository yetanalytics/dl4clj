(ns dl4clj.datasets.iterator.iterators
  (:import [org.deeplearning4j.datasets.iterator
            DoublesDataSetIterator FloatsDataSetIterator INDArrayDataSetIterator
            AsyncDataSetIterator AsyncMultiDataSetIterator CombinedPreProcessor
            CombinedPreProcessor$Builder CurvesDataSetIterator IteratorDataSetIterator
            IteratorMultiDataSetIterator MovingWindowBaseDataSetIterator
            MultipleEpochsIterator ReconstructionDataSetIterator SamplingDataSetIterator
            ExistingDataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MultiDataSetIteratorAdapter
            ListDataSetIterator SingletonMultiDataSetIterator])
  (:require [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]
            [dl4clj.berkeley :refer [new-pair]]
            [dl4clj.helpers :refer :all]))


;; figure out where to put the dataset pre-processors
;; prob in the same namespace as normalizers

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-combined-pre-processor
  "This is special preProcessor, that allows to combine multiple prerpocessors,
  and apply them to data sequentially.

   pre-processors (map), {(int) (pre-processor)
                         (int) (pre-processor)}

   - the keys are the desired indexes of the pre-processors (dataset index within the iterator)
   - values are the pre-processors themselves
   - pre-processors should be fit to a dataset or iterator before being combined

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/CombinedPreProcessor.html"
  [pre-processors]
  (loop [b (CombinedPreProcessor$Builder.)
         result pre-processors]
    (if (empty? result)
      (.build b)
      (let [data (first result)
            [idx pp] data]
        (recur (doto b (.addPreProcessor idx pp)) (rest result))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api calls
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn pre-process-iter-combined-pp!
  "Pre process a dataset sequentially using a combined pre-processor
   - the pre-processor is attached to the dataset"
  [& {:keys [iter dataset]}]
  (let [ds-iter (reset-if-empty?! iter)]
   (doto ds-iter (.preProcess dataset))))
