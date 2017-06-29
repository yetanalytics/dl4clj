(ns dl4clj.datasets.pre-processors
  (:import [org.nd4j.linalg.dataset.api.preprocessor
            ImageFlatteningDataSetPreProcessor
            ImagePreProcessingScaler
            NormalizerMinMaxScaler
            NormalizerStandardize
            VGG16ImagePreProcessor]
           [org.deeplearning4j.datasets.iterator
            CombinedPreProcessor CombinedPreProcessor$Builder])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many? array-of]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

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

(defmethod pre-processors :standardize-normalization [opts]
  (let [conf (:standardize-normalization opts)
        {f-mean :features-mean
         f-std :features-std
         l-mean :labels-mean
         l-std :labels-std} conf]
    (let [f-m (vec-or-matrix->indarray f-mean)
          f-s (vec-or-matrix->indarray f-std)]
     (cond (contains-many? conf :features-mean :features-std :labels-mean :labels-std)
           (NormalizerStandardize. f-m f-s (vec-or-matrix->indarray l-mean)
                                   (vec-or-matrix->indarray l-std))
          (contains-many? conf :features-mean :features-std)
          (NormalizerStandardize. f-m f-s)
          :else
          (NormalizerStandardize.)))))

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

  :features-mean (vec or INDArray), the features to normalize

  :features-std (vec or INDArray), the features to normalize

  :labels-mean (vec or INDArray), the labels to normalize

  :labels-std (vec or INDArray), the labels to normalize"
  [& {:keys [features-mean features-std labels-mean labels-std]
      :as opts}]
  (pre-processors {:standardize-normalization opts}))


(defn new-vgg16-image-preprocessor
  "This is a preprocessor specifically for VGG16.
  It subtracts the mean RGB value, computed on the training set, from each pixel"
  []
  (pre-processors {:vgg16 {}}))

(defn new-combined-pre-processor
  "This is special preProcessor, that allows to combine multiple prerpocessors,
  and apply them to data sequentially.

   pre-processors (map), {(int) (pre-processor)
                         (int) (pre-processor-config-map)}

   - the keys are the desired indexes of the pre-processors (dataset index within the iterator)
   - values are the pre-processors themselves or configuration maps for creating pre-processors
   - pre-processors should be fit to a dataset or iterator before being combined (double check on this)
     - if this is the case, config maps dont make sense

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/CombinedPreProcessor.html"
  [map-of-pre-processors]
  (loop [b (CombinedPreProcessor$Builder.)
         result map-of-pre-processors]
    (if (empty? result)
      (.build b)
      (let [data (first result)
            [idx pp] data]
        (recur (doto b (.addPreProcessor idx (if (map? pp)
                                               (pre-processors pp)
                                               pp)))
               (rest result))))))
