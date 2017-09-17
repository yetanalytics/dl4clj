(ns dl4clj.nn.api.fine-tune
  (:import [org.deeplearning4j.nn.transferlearning
            FineTuneConfiguration TransferLearningHelper]
           [org.nd4j.linalg.api.ndarray INDArray])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [array-of obj-or-code?]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.helpers :refer [reset-iterator!]]))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fine tune confs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn applied-to-nn-conf!
  "applies a fine tune configuration to a supplied neural network configuration.

   Returns the mutated nn-conf"
  [& {:keys [fine-tune-conf nn-conf as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:fine-tune-conf (_ :guard seq?)
           :nn-conf (_ :guard seq?)}]
         (obj-or-code? as-code? `(.appliedNeuralNetConfiguration ~fine-tune-conf ~nn-conf))
         :else
         (.appliedNeuralNetConfiguration fine-tune-conf nn-conf)))

(defn nn-conf-from-fine-tune-conf
  "creates a neural network configuration builder from a fine tune configuration.

  the resulting nn-conf has the fine-tune-confs opts applied.

  :build? (boolean), determines if a nn-conf builder or nn-conf is returned"
  [& {:keys [fine-tune-conf build? as-code?]
      :or {build? true
           as-code? true}}]
  (let [m-call (match [fine-tune-conf]
                      [(_ :guard seq?)]
                      `(.appliedNeuralNetConfigurationBuilder ~fine-tune-conf)
                      :else
                      (.appliedNeuralNetConfigurationBuilder fine-tune-conf))]
    (cond (and build? (seq? fine-tune-conf))
          (obj-or-code? as-code? `(.build ~m-call))
          build?
          (.build m-call)
          :else
          (obj-or-code? as-code? m-call))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fine tune helpers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn featurize
  "During training frozen vertices/layers can be treated as featurizing the input
  The forward pass through these frozen layer/vertices can be done in advance
  and the dataset saved to disk to iterate quickly on the smaller unfrozen
  part of the model Currently does not support datasets with feature masks

  :helper (TransferLearningHelper), created by new-helper

  :data-set (dataset or multi-dataset), a dataset
   - can be a single or multi dataset
   - see: nd4clj.linalg.dataset.data-set (under construction)

  warning, this can crash if the dataset is too large"
  [& {:keys [helper data-set as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:helper (_ :guard seq?)
           :data-set (_ :guard seq?)}]
         (obj-or-code? as-code? `(.featurize ~helper ~data-set))
         :else
         (.featurize helper data-set)))

(defn fit-featurized!
  "Fit from a featurized dataset

  :helper (TransferLearningHelper), created by new-helper

  :data-set (dataset or multi-dataset), a dataset
   - can be a single or multi dataset
   - see: nd4clj.linalg.dataset.data-set (under construction)

  :iter (dataset-iterator or multi-dataset-iterator) a ds iterator
   - can be built based on a single or multi dataset
   - see: dl4clj.datasets.iterator.iterators (double check this through testing)

  returns the helper"
  [& {:keys [helper data-set iter as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:helper (_ :guard seq?) :data-set (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~helper (.fitFeaturized ~data-set)))
         [{:helper _ :data-set _}]
         (doto helper (.fitFeaturized data-set))
         [{:helper (_ :guard seq?) :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~helper (.fitFeaturized ~iter)))
         [{:helper _ :iter _}]
         (doto helper (.fitFeaturized (reset-iterator! iter)))
         :else
         (assert false "you must supply either a data-set or a dat-set iterator")))

(defn output-from-featurized
  "Use to get the output from a featurized input

  :helper (TransferLearningHelper), created by new-helper

  :featurized-input (INDArray or vec), featurized data

  :array-of-featurized-input (coll of INDArrays or the code to create them),
   - multiple featurized inputs"
  [& {:keys [helper featurized-input array-of-featurized-input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:helper (_ :guard seq?)
           :array-of-featurized-input _}]
         (obj-or-code?
          as-code?
          `(.outputFromFeaturized ~helper (array-of :data ~array-of-featurized-input
                                                   :java-type INDArray)))
         [{:helper _
           :array-of-featurized-input _}]
         (.outputFromFeaturized helper (array-of :data array-of-featurized-input
                                                 :java-type INDArray))
         [{:helper (_ :guard seq?)
           :featurized-input (:or (_ :guard vector?)
                                  (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.outputFromFeaturized ~helper (vec-or-matrix->indarray ~featurized-input)))
         [{:helper _
           :featurized-input _}]
         (.outputFromFeaturized helper (vec-or-matrix->indarray featurized-input))))

(defn unfrozen-mln
  "Returns the unfrozen layers of the MultiLayerNetwork as a multilayernetwork
   Note that with each call to featurizedFit the parameters to the original MLN are also updated

  need to test if this returns the mutated og network with all layers or only the frozen layers
   - if its just the previously frozen layers, will need to merge back into og model
     - og model may have been mutated"
  [& {:keys [helper as-code?]
      :or {as-code? true}}]
  (match [helper]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.unfrozenMLN ~helper))
         :else
         (.unfrozenMLN helper)))
