(ns ^{:doc "Pre process a dataset, see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/DataSetPreProcessor.html

Base interface for all normalizers see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/Normalizer.html

An interface for data normalizers. Data normalizers compute some sort of statistics over a dataset and scale the data in some way.
see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/DataNormalization.html"}
    dl4clj.datasets.api.pre-processors
  (:import [org.nd4j.linalg.dataset.api DataSetPreProcessor]
           [org.nd4j.linalg.dataset.api.preprocessor Normalizer
            DataNormalization]
           [org.nd4j.linalg.dataset.api.preprocessor
            ImageFlatteningDataSetPreProcessor
            ImagePreProcessingScaler
            NormalizerMinMaxScaler
            NormalizerStandardize
            VGG16ImagePreProcessor]
           [org.deeplearning4j.datasets.iterator CombinedPreProcessor])
  (:require [dl4clj.helpers :refer [reset-if-empty?!]]))

(defn pre-process-dataset!
  "Pre process a dataset

  returns a the dataset"
  [& {:keys [pre-processor ds]}]
  (.preProcess pre-processor ds)
  ds)

(defn pre-process-iter-combined-pp!
  "Pre process a dataset sequentially using a combined pre-processor
   - the pre-processor is attached to the dataset

  returns the iterator for the dataset"
  [& {:keys [iter dataset]}]
  (let [ds-iter (reset-if-empty?! iter)]
    (doto ds-iter (.preProcess dataset))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; generic normalizer (pre-processor) fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-normalizer-type
  "Get the enum type of this normalizer"
  [normalizer]
  (.getType normalizer))

(defn fit-iter!
  "Iterates over a dataset accumulating statistics for normalization

  returns the fit normalizer"
  [& {:keys [normalizer iter]}]
  (doto normalizer (.fit (reset-if-empty?! iter))))

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
    (.revertFeatures normalizer features)))

(defn revert-labels!
  "Undo the normalization applied by the normalizer on the labels array

  :labels (INDArray), nn targets

  :labels-mask (INDArray), mask for the nn-targets

  returns the (un)normalized labels"
  [& {:keys [normalizer labels labels-mask]
      :as opts}]
  (if (contains? opts :labels-mask)
    (.revertLabels normalizer labels labels-mask)
    (.revertLabels normalizer labels)))

(defn transform-features!
  "applies the transform specified by the normalizer to the features

  :features (INDArray), nn input

  :features-mask (INDArray), mask for the nn-input

  returns the normalized features"
  [& {:keys [normalizer features features-mask]
      :as opts}]
  (if (contains? opts :features-mask)
    (.transform normalizer features features-mask)
    (.transform normalizer features)))

(defn transform-labels!
  "applies the transform specified by the normalizer to the labels

  :labels (INDArray), nn targets

  :labels-mask (INDArray), mask for the nn-targets

  returns the normalized labels"
  [& {:keys [normalizer labels labels-mask]
      :as opts}]
  (if (contains? opts :labels-mask)
    (.transformLabel normalizer labels labels-mask)
    (.transformLabel normalizer labels)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min-max-normalization specific fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-labels-max
  [min-max-pp]
  (.getLabelMax min-max-pp))

(defn get-labels-min
  [min-max-pp]
  (.getLabelMin min-max-pp))

(defn get-max
  [min-max-pp]
  (.getMax min-max-pp))

(defn get-min
  [min-max-pp]
  (.getMin min-max-pp))

(defn get-target-max
  [min-max-pp]
  (.getTargetMax min-max-pp))

(defn get-target-min
  [min-max-pp]
  (.getTargetMin min-max-pp))

(defn load-min-max
  "Load the given min and max form the supplied file(s)

  :files (coll), collection of file paths to be loaded

  :pp (pre-processor). can be the standardizer or the min-max"
  [& {:keys [pp files]}]
  (.load pp (array-of :data (map clojure.java.io/as-file files)
                      :java-type java.io.File)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; standardize specific fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-label-std
  [standardize-pp]
  (.getLabelStd standardize-pp))

(defn get-mean
  [standardize-pp]
  (.getMean standardize-pp))

(defn get-std
  [standardize-pp]
  (.getStd standardize-pp))

(defn get-label-mean
  [standardize-pp]
  (.getLabelMean standardize-pp))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vgg16 specific fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn preprocess-features!
  "preprcess the features

  :features (INDArray), the features

  returns the processed features"
  [& {:keys [vgg16-pp features]}]
  (.preProcess vgg16-pp features))
