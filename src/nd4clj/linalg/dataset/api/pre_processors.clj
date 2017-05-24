(ns nd4clj.linalg.dataset.api.pre-processors
  (:import [org.nd4j.linalg.dataset.api.preprocessor
            ImageFlatteningDataSetPreProcessor
            ImagePreProcessingScaler
            NormalizerMinMaxScaler
            NormalizerStandardize
            VGG16ImagePreProcessor])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many? array-of]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; pre-processer creation multimethod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti pre-processors generic-dispatching-fn)

(defmethod pre-processors :image-flattening [opts]
  (ImageFlatteningDataSetPreProcessor.))

(defmethod pre-processors :vgg16 [opts]
  (VGG16ImagePreProcessor.))

(defmethod pre-processors :image-scaling [opts]
  (let [conf (:image-scaling opts)
        {a :min-range
         b :max-range
         max-bits :max-bits} conf]
    (cond (contains-many? conf :min-range :max-range :max-bites)
          (ImagePreProcessingScaler. a b max-bits)
          (contains-many? conf :min-range :max-range)
          (ImagePreProcessingScaler. a b)
          :else
          (ImagePreProcessingScaler.))))

(defmethod pre-processors :min-max-normalization [opts]
  (let [conf (:min-max-normalization opts)
        {a :min-val
         b :max-val} conf]
    (if (contains-many? conf :min-val :max-val)
      (NormalizerMinMaxScaler. a b)
      (NormalizerMinMaxScaler.))))

(defmethod pre-processors :standardize [opts]
  (let [conf (:standardize opts)
        {f-mean :features-mean
         f-std :features-std
         l-mean :labels-mean
         l-std :labels-std} conf]
    (cond (contains-many? conf :features-mean :features-std :labels-mean :labels-std)
          (NormalizerStandardize. f-mean f-std l-mean l-std)
          (contains-many? conf :features-mean :features-std)
          (NormalizerStandardize. f-mean f-std)
          :else
          (NormalizerStandardize.))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; pre-processer creation user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-image-flattening-ds-preprocessor
  "A DataSetPreProcessor used to flatten a 4d CNN features array to a flattened 2d format
    - for use in dense-layers and multi-layer perceptrons"
  []
  (pre-processors {:image-flattening {}}))

(defn new-image-scaling-ds-preprocessor
  "A preprocessor specifically for images that applies min max scaling
   - ie. pixel values can be scaled from 0->255 to :min-range->:max-range
   - defaults to :min-range = 0, :max-range = 1

   :min-range (double), the low end of the scale

   :max-range (double), the high end of the scale

   :max-bits (int), If pixel values are not 8 bits, set the bits value here
    - For values that are already floating point, specify the number of bits as 1"
  [& {:keys [min-range max-range max-bits]
      :as opts}]
  (pre-processors {:image-scaling opts}))

(defn new-min-max-normalization-ds-preprocessor
  "Pre processor for DataSets that normalizes feature values (and optionally label values)
  to lie between :min-val and :max-val
   - defaults to 0 -> 1

  :min-val (double), the low end of the scale

  :max-val (double), the high end of the scale"
  [& {:keys [min-val max-val]
      :as opts}]
  (pre-processors {:min-max-normalization opts}))

(defn new-standardize-normalization-ds-preprocessor
  "ormalizes feature values (and optionally label values)
  to have a 0 mean and a standard deviation of 1

  :features-mean (INDArray), the features to normalize

  :features-std (INDArray), the features to normalize

  :labels-mean (INDArray), the labels to normalize

  :labels-std (INDArray), the labels to normalize"
  [& {:keys [features-mean features-std labels-mean labels-std]
      :as opts}]
  (pre-processors {:standardize opts}))


(defn new-vgg16-image-preprocessor
  "This is a preprocessor specifically for VGG16.
  It subtracts the mean RGB value, computed on the training set, from each pixel"
  []
  (pre-processors {:vgg16 {}}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; pre-processer specific fns
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

(defn get-label-mean
  [standardize-pp]
  (.getLabelMean standardize-pp))

(defn get-label-std
  [standardize-pp]
  (.getLabelStd standardize-pp))

(defn get-mean
  [standardize-pp]
  (.getMean standardize-pp))

(defn get-std
  [standardize-pp]
  (.getStd standardize-pp))

(defn load-min-max
  "Load the given min and max form the supplied file(s)

  :files (coll), collection of file paths to be loaded

  :pp (pre-processor). can be the standardizer or the min-max"
  [& {:keys [pp files]}]
  (.load pp (array-of :data (map clojure.java.io/as-file files)
                              :java-type java.io.File)))

(defn preprocess-features!
  "preprcess the features

  :features (INDArray), the features

  returns the processed features"
  [& {:keys [vgg16-pp features]}]
  (.preProcess vgg16-pp features)
  features)
