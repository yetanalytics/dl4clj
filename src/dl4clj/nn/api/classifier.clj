(ns ^{:doc "Implementation of the methods found in the Classifier Interface.
fns are for classification models (this is for supervised learning)
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Classifier.html"}
  dl4clj.nn.api.classifier
  (:import [org.deeplearning4j.nn.api Classifier])
  (:require [dl4clj.utils :refer [contains-many? obj-or-code? eval-if-code]]
            [dl4clj.helpers :refer [reset-iterator!]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn f1-score
  "With two arguments (classifier and dataset):
   - Sets the input and labels and returns a score for the prediction.

  With three arguments (classifier, examples and labels):
   - Returns the f1 score for the given examples.
   - examples and labels should both be INDArrays or vectors
    - examples = the data you want to classify
    - labels = the correct classifcation for a a set of examples"
  [& {:keys [classifier dataset examples labels as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:classifier (_ :guard seq?)
           :dataset (_ :guard seq?)}]
         (obj-or-code? as-code? `(.f1Score ~classifier ~dataset))
         [{:classifier _ :dataset _}]
         (let [[c-obj ds-obj] (eval-if-code [classifier seq?] [dataset seq?])]
           (.f1Score c-obj ds-obj))
         [{:classifier (_ :guard seq?)
           :examples (:or (_ :guard vector?)
                          (_ :guard seq?))
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.f1Score ~classifier
                    (vec-or-matrix->indarray ~examples)
                    (vec-or-matrix->indarray ~labels)))
         [{:classifier _ :examples _ :labels _}]
         (let [[c-obj ex-vec l-vec] (eval-if-code [classifier seq?]
                                                  [examples seq?]
                                                  [labels seq?])]
           (.f1Score c-obj
                     (vec-or-matrix->indarray ex-vec)
                     (vec-or-matrix->indarray l-vec)))
         :else
         (assert false "you must supply a classifier and either a dataset or
examples and their labels")))

(defn label-probabilities
  "Returns the probabilities for each label for each example row wise

  :examples (INDArray or vec), the examples to classify (one example in each row)"
  [& {:keys [classifier examples as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:classifier (_ :guard seq?)
           :examples (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.labelProbabilities ~classifier (vec-or-matrix->indarray ~examples)))
         [{:classifier _ :examples _}]
         (let [[c-obj ex-vec] (eval-if-code [classifier seq?]
                                            [examples seq?])]
           (.labelProbabilities c-obj (vec-or-matrix->indarray ex-vec)))))

(defn num-labels
  "Returns the number of possible labels"
  [classifier & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [classifier]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.numLabels ~classifier))
         :else
         (.numLabels classifier)))

(defn predict
  "Takes in a list of examples for each row (INDArray or vec), returns a label

   or

  takes a datset of examples for each row, returns a label"
  [& {:keys [classifier examples dataset as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:classifier (_ :guard seq?)
           :examples (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(.predict ~classifier (vec-or-matrix->indarray ~examples)))
         [{:classifier _ :examples _}]
         (let [[c-obj ex-vec] (eval-if-code [classifier seq?]
                                            [examples seq?])]
           (.predict c-obj (vec-or-matrix->indarray ex-vec)))
         [{:classifier (_ :guard seq?) :dataset (_ :guard seq?)}]
         (obj-or-code? as-code? `(.predict ~classifier ~dataset))
         [{:classifier _ :dataset _}]
         (let [[c-obj ds-obj] (eval-if-code [classifier seq?]
                                            [dataset seq?])]
           (.predict c-obj ds-obj))
         :else
         (assert false "you must supply a classifier and either an INDArray of examples or a dataset")))
