(ns dl4clj.nn.api.fine-tune
  (:import [org.deeplearning4j.nn.transferlearning
            FineTuneConfiguration TransferLearningHelper]
           [org.nd4j.linalg.api.ndarray INDArray])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [array-of]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.helpers :refer [reset-iterator!]]))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fine tune confs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn applied-to-nn-conf!
  "applies a fine tune configuration to a supplied neural network configuration.

   Returns the mutated nn-conf"
  [& {:keys [fine-tune-conf nn-conf]}]
  (.appliedNeuralNetConfiguration fine-tune-conf nn-conf))

(defn nn-conf-from-fine-tune-conf
  "creates a neural network configuration builder from a fine tune configuration.

  the resulting nn-conf has the fine-tune-confs opts applied.

  :build? (boolean), determines if a nn-conf builder or nn-conf is returned"
  [& {:keys [fine-tune-conf build?]
      :or {build? false}}]
  (if build?
    (.build (.appliedNeuralNetConfigurationBuilder fine-tune-conf))
    (.appliedNeuralNetConfigurationBuilder fine-tune-conf)))

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
  [& {:keys [helper data-set]}]
  (.featurize helper data-set))

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
  [& {:keys [helper data-set iter]
   :as opts}]
  (match [opts]
         [{:helper _ :data-set _}] (doto helper (.fitFeaturized data-set))
         [{:helper _ :iter _}] (doto helper (.fitFeaturized (reset-iterator! iter)))
         :else
         (assert false "you must supply either a data-set or a dat-set iterator")))

(defn output-from-featurized
  "Use to get the output from a featurized input

  :helper (TransferLearningHelper), created by new-helper

  :featurized-input (INDArray or vec), featurized data

  :array-of-featurized-input (coll of INDArrays), multiple featurized inputs"
  [& {:keys [helper featurized-input array-of-featurized-input]
      :as opts}]
  (if array-of-featurized-input
    (.outputFromFeaturized helper (array-of :data array-of-featurized-input
                                            :java-type INDArray))
    (.outputFromFeaturized helper (vec-or-matrix->indarray featurized-input))))

(defn unfrozen-mln
  "Returns the unfrozen layers of the MultiLayerNetwork as a multilayernetwork
   Note that with each call to featurizedFit the parameters to the original MLN are also updated

  need to test if this returns the mutated og network with all layers or only the frozen layers
   - if its just the previously frozen layers, will need to merge back into og model
     - og model may have been mutated"
  [helper]
  (.unfrozenMLN helper))
