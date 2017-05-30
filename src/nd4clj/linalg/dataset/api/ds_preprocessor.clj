(ns ^{:doc "Pre process a dataset, see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/DataSetPreProcessor.html

Base interface for all normalizers see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/Normalizer.html

An interface for data normalizers. Data normalizers compute some sort of statistics over a dataset and scale the data in some way.
see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/DataNormalization.html"}
    nd4clj.linalg.dataset.api.ds-preprocessor
  (:import [org.nd4j.linalg.dataset.api DataSetPreProcessor]
           [org.nd4j.linalg.dataset.api.preprocessor Normalizer
            DataNormalization]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; data-set-pre-processor interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn pre-process-dataset!
  "Pre process a dataset

  returns the dataset"
  [& {:keys [pre-processor ds]}]
  (.preProcess pre-processor ds)
  ds)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; normalizer interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn fit-dataset!
  "Fit a dataset (only compute based on the statistics from this dataset)"
  [& {:keys [normalizer ds]}]
  (.fit normalizer ds))

(defn get-normalizer-type
  "Get the enum type of this normalizer"
  [normalizer]
  (.getType normalizer))

(defn revert-normalization!
  "undo the normalization applied by the normalizer.

  returns the now (un)normalized dataset"
  [& {:keys [normalizer normalized]}]
  (.revert normalizer normalized)
  normalized)

(defn transform-dataset!
  "transforms the dataset and returns it"
  [& {:keys [normalizer ds]}]
  (.transform normalizer ds)
  ds)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; data normalization interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn fit-iter!
  "Iterates over a dataset accumulating statistics for normalization

  returns the normalizer"
  [& {:keys [normalizer ds-iter]}]
  (doto normalizer (.fit ds-iter)))

(defn fit-labels!?
  "Flag to specify if the labels/outputs in the dataset should be also normalized.

  returns the normalizer"
  [& {:keys [normalizer fit-labels?]}]
  (doto normalizer (.fitLabel fit-labels?)))

(defn normalize-labels?
  "Whether normalization for the labels is also enabled."
  [normalizer]
  (.isFitLabel normalizer))

(defn revert-features!
  "Undo the normalization applied by the normalizer on the features array

  :features (INDArray), nn input

  :features-mask (INDArray), mask for the nn-input

  returns the (un)normalized features"
  [& {:keys [normalizer features features-mask]
      :as opts}]
  (if (contains? opts :features-mask)
    (.revertFeatures normalizer features features-mask)
    (.revertFeatures normalizer features))
  features)

(defn revert-labels!
  "Undo the normalization applied by the normalizer on the labels array

  :labels (INDArray), nn targets

  :labels-mask (INDArray), mask for the nn-targets

  returns the (un)normalized labels"
  [& {:keys [normalizer labels labels-mask]
      :as opts}]
  (if (contains? opts :labels-mask)
    (.revertLabels normalizer labels labels-mask)
    (.revertLabels normalizer labels))
  labels)

(defn transform-features!
  "applies the transform specified by the normalizer to the features

  :features (INDArray), nn input

  :features-mask (INDArray), mask for the nn-input

  returns the normalized features"
  [& {:keys [normalizer features features-mask]
      :as opts}]
  (if (contains? opts :features-mask)
    (.transform normalizer features features-mask)
    (.transform normalizer features))
  features)

(defn transform-labels!
  "applies the transform specified by the normalizer to the labels

  :labels (INDArray), nn targets

  :labels-mask (INDArray), mask for the nn-targets

  returns the normalized labels"
  [& {:keys [normalizer labels labels-mask]
      :as opts}]
  (if (contains? opts :labels-mask)
    (.transformLabel normalizer labels labels-mask)
    (.transformLabel normalizer labels))
  labels)
