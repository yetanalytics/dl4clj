(ns dl4clj.datasets.pre-processors
  (:import [org.nd4j.linalg.dataset.api.preprocessor
            ImageFlatteningDataSetPreProcessor
            ImagePreProcessingScaler
            NormalizerMinMaxScaler
            NormalizerStandardize
            VGG16ImagePreProcessor]
           [org.deeplearning4j.datasets.iterator
            CombinedPreProcessor CombinedPreProcessor$Builder])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many? array-of
                                  builder-fn eval-and-build]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;; replace contains-many? with core.match
(defmulti pre-processors generic-dispatching-fn)

(defmethod pre-processors :image-flattening [opts]
  `(ImageFlatteningDataSetPreProcessor.))

(defmethod pre-processors :vgg16 [opts]
  `(VGG16ImagePreProcessor.))

(defmethod pre-processors :image-scaling [opts]
  (let [conf (:image-scaling opts)
        {a :min-range
         b :max-range
         max-bits :max-bits} conf]
    (match [conf]
           [{:min-range _ :max-range _ :max-bits _}]
           `(ImagePreProcessingScaler. ~a ~b ~max-bits)
           [{:min-range _ :max-range _}]
           `(ImagePreProcessingScaler. ~a ~b)
           :else
           `(ImagePreProcessingScaler.))))

(defmethod pre-processors :min-max-normalization [opts]
  (let [conf (:min-max-normalization opts)
        {a :min-val
         b :max-val} conf]
    (if (contains-many? conf :min-val :max-val)
      `(NormalizerMinMaxScaler. ~a ~b)
      `(NormalizerMinMaxScaler.))))

(defmethod pre-processors :standardize-normalization [opts]
  (let [conf (:standardize-normalization opts)
        {f-mean :features-mean
         f-std :features-std
         l-mean :labels-mean
         l-std :labels-std} conf]
    (let [f-m (vec-or-matrix->indarray f-mean)
          f-s (vec-or-matrix->indarray f-std)]
      (match [conf]
             [{:features-mean _ :features-std _
               :labels-mean _ :labels-std _}]
             `(NormalizerStandardize. ~f-m ~f-s
                                      (vec-or-matrix->indarray ~l-mean)
                                     (vec-or-matrix->indarray ~l-std))
             [{:features-mean _ :features-std _}]
             `(NormalizerStandardize. ~f-m ~f-s)
             :else
             `(NormalizerStandardize.)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; pre-processer creation user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-image-flattening-ds-preprocessor
  "A DataSetPreProcessor used to flatten a 4d CNN features array to a flattened 2d format
    - for use in dense-layers and multi-layer perceptrons

  :as-code? (boolean), return java object or code for creating it"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:image-flattening {}})]
    (if as-code?
      code
      (eval code))))

(defn new-image-scaling-ds-preprocessor
  "A preprocessor specifically for images that applies min max scaling
   - ie. pixel values can be scaled from 0->255 to :min-range->:max-range
   - defaults to :min-range = 0, :max-range = 1

   :min-range (double), the low end of the scale

   :max-range (double), the high end of the scale

   :max-bits (int), If pixel values are not 8 bits, set the bits value here
    - For values that are already floating point, specify the number of bits as 1

   :as-code? (boolean), return java object or code for creating it"
  [& {:keys [min-range max-range max-bits as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (pre-processors {:image-scaling opts})]
    (if as-code?
      code
      (eval code))))

(defn new-min-max-normalization-ds-preprocessor
  "Pre processor for DataSets that normalizes feature values (and optionally label values)
  to lie between :min-val and :max-val
   - defaults to 0 -> 1

  :min-val (double), the low end of the scale

  :max-val (double), the high end of the scale

  :as-code? (boolean), return java object or code for creating it

  WARNING, there is something misunderstood about this preprocessor.
   - come back and investigate why the max range and min range are not being properly set"
  [& {:keys [min-val max-val as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (pre-processors {:min-max-normalization opts})]
    (if as-code?
      code
      (eval code))))

(defn new-standardize-normalization-ds-preprocessor
  "ormalizes feature values (and optionally label values)
  to have a 0 mean and a standard deviation of 1

  :features-mean (vec or INDArray), the features to normalize

  :features-std (vec or INDArray), the features to normalize

  :labels-mean (vec or INDArray), the labels to normalize

  :labels-std (vec or INDArray), the labels to normalize

  :as-code? (boolean), return java object or code for creating it"
  [& {:keys [features-mean features-std labels-mean labels-std as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (pre-processors {:standardize-normalization opts})]
    (if as-code?
      code
      (eval code))))


(defn new-vgg16-image-preprocessor
  "This is a preprocessor specifically for VGG16.
  It subtracts the mean RGB value, computed on the training set, from each pixel

  :as-code? (boolean), return java object or code for creating it"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (pre-processors {:vgg16 {}})]
    (if as-code?
      code
      (eval code))))

(defn new-combined-pre-processor
  "This is special preProcessor, that allows to combine multiple prerpocessors,
  and apply them to data sequentially.

   pre-processor (map), {(int) (pre-processor)
                         (int) (pre-processor-config-map)}

   - the keys are the desired indexes of the pre-processors (dataset index within the iterator)
   - values are the pre-processors themselves or configuration maps for creating pre-processors
   - pre-processors should be fit to a dataset or iterator before being combined (double check on this)
     - if this is the case, config maps dont make sense

   :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/CombinedPreProcessor.html"
  [& {:keys [pre-processor as-code?]
      :or {as-code? true}}]
  (let [code (builder-fn `(CombinedPreProcessor$Builder.)
                         {:add '.addPreProcessor}
                         {:add (into [] (for [each pre-processor
                                              :let [[idx pp] each]]
                                          (match [pp]
                                                 [(_ :guard seq?)] [idx pp]
                                                 :else
                                                 [idx (pre-processors pp)])))})]
    (if as-code?
      `(.build ~code)
      (eval-and-build code))))
