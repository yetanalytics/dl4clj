(ns nd4clj.linalg.dataset.multi-ds
  (:import [org.nd4j.linalg.dataset MultiDataSet])
  (:require [dl4clj.utils :refer [contains-many?]]))

;; come back and create the interface for multidatasets

(defn new-multi-ds
  "a dataset that contains multiple datasets"
  ;; come back and beef up this doc string
  ;; also ensure the arrays  are INDArrays of INDarrays
  [& {:keys [features labels features-mask labels-mask]
      :as opts}]
  (cond (contains-many? opts :features :labels :features-mask :labels-mask)
        (MultiDataSet. features labels features-mask labels-mask)
        (contains-many? opts :features :labels)
        ;; bc the constructor can accept these as just a single INDArray
        ;; should throw in a conditional to account for this
        (MultiDataSet. features labels)
        :else
        (MultiDataSet.)))
