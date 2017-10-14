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
  (:require [dl4clj.helpers :refer [reset-iterator!]]
            [dl4clj.utils :refer [array-of obj-or-code?]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;; TODO: make pre-processor/normalizer language consistent

(defn pre-process-dataset!
  "Pre process a dataset

  returns a the dataset"
  [& {:keys [pre-processor ds as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pre-processor (_ :guard seq?)
           :ds (_ :guard seq?)}]
         (obj-or-code? as-code? `(do (.preProcess ~pre-processor ~ds)
                                     ~ds))
         :else
         (do (.preProcess pre-processor ds)
             ds)))

(defn pre-process-iter-combined-pp!
  "Pre process a dataset sequentially using a combined pre-processor
   - the pre-processor is attached to the dataset

  returns the iterator for the dataset"
  [& {:keys [iter dataset as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :dataset (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~iter (.preProcess ~dataset)))
         :else
         (doto (reset-iterator! iter) (.preProcess dataset))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; generic normalizer (pre-processor) fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-normalizer-type
  "Get the enum type of this normalizer"
  [normalizer & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [normalizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getType ~normalizer))
         :else
         (.getType normalizer)))

(defn fit-iter!
  "Iterates over a dataset accumulating statistics for normalization

  returns the fit normalizer"
  [& {:keys [normalizer iter as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normalizer (_ :guard seq?)
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~normalizer (.fit ~iter)))
         :else
         (doto normalizer (.fit iter))))

(defn fit-labels!?
  "Flag to specify if the labels/outputs in the dataset should be also normalized.

  returns the normalizer"
  [& {:keys [normalizer fit-labels? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normalizer (_ :guard seq?)
           :fit-labels (:or (_ :guard boolean?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~normalizer (.fitLabel ~fit-labels?)))
         :else
         (doto normalizer (.fitLabel fit-labels?))))

(defn normalize-labels?
  "Whether normalization for the labels is also enabled."
  [normalizer & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [normalizer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.isFitLabel ~normalizer))
         :else
         (.isFitLabel normalizer)))

(defn revert-features!
  "Undo the normalization applied by the normalizer on the features array

  :features (vec or INDArray), nn input

  :features-mask (vec or INDArray), mask for the nn-input

  returns the (un)normalized features"
  [& {:keys [normalizer features features-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normalizer (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :features-mask (:or (_ :guard vector?)
                               (_ :guard seq?))}]
         (obj-or-code? as-code? `(.revertFeatures ~normalizer (vec-or-matrix->indarray ~features)
                                                  (vec-or-matrix->indarray ~features-mask)))
         [{:normalizer _
           :features _
           :features-mask _}]
         (.revertFeatures normalizer (vec-or-matrix->indarray features)
                          (vec-or-matrix->indarray features-mask))
         [{:normalizer (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(.revertFeatures ~normalizer (vec-or-matrix->indarray ~features)))
         [{:normalizer _
           :features _}]
         (.revertFeatures normalizer (vec-or-matrix->indarray features))))

(defn revert-labels!
  "Undo the normalization applied by the normalizer on the labels array

  :labels (vec or INDArray), nn targets

  :labels-mask (vec or INDArray), mask for the nn-targets

  returns the (un)normalized labels"
  [& {:keys [normalizer labels labels-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normalizer (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.revertLabels ~normalizer (vec-or-matrix->indarray ~labels)
                          (vec-or-matrix->indarray ~labels-mask)))
         [{:normalizer _
           :labels _
           :labels-mask _}]
         (.revertLabels normalizer (vec-or-matrix->indarray labels)
                        (vec-or-matrix->indarray labels-mask))
         [{:normalizer (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(.revertLabels ~normalizer (vec-or-matrix->indarray ~labels)))
         [{:normalizer _
           :labels _}]
         (.revertLabels normalizer (vec-or-matrix->indarray labels))))

(defn transform-features!
  "applies the transform specified by the normalizer to the features

  :features (vec or INDArray), nn input

  :features-mask (vec or INDArray), mask for the nn-input

  returns the normalized features"
  [& {:keys [normalizer features features-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normalizer (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :features-mask (:or (_ :guard vector?)
                               (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transform ~normalizer (vec-or-matrix->indarray ~features)
                       (vec-or-matrix->indarray ~features-mask)))
         [{:normalizer _
           :features _
           :features-mask _}]
         (.transform normalizer (vec-or-matrix->indarray features)
                     (vec-or-matrix->indarray features-mask))
         [{:normalizer (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transform ~normalizer (vec-or-matrix->indarray ~features)))
         [{:normalizer _
           :features _}]
         (.transform normalizer (vec-or-matrix->indarray features))))

(defn transform-labels!
  "applies the transform specified by the normalizer to the labels

  :labels (vec or INDArray), nn targets

  :labels-mask (vec or INDArray), mask for the nn-targets

  returns the normalized labels"
  [& {:keys [normalizer labels labels-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normalizer (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transformLabel ~normalizer (vec-or-matrix->indarray ~labels)
                            (vec-or-matrix->indarray ~labels-mask)))
         [{:normalizer _
           :labels _
           :labels-mask _}]
         (.transformLabel normalizer (vec-or-matrix->indarray labels)
                          (vec-or-matrix->indarray labels-mask))
         [{:normalizer (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transformLabel ~normalizer (vec-or-matrix->indarray ~labels)))
         [{:normalizer _
           :labels _}]
         (.transformLabel normalizer (vec-or-matrix->indarray labels))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min-max-normalization specific fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-labels-max
  [min-max-pp & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [min-max-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLabelMax ~min-max-pp))
         :else
         (.getLabelMax min-max-pp)))

(defn get-labels-min
  [min-max-pp & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [min-max-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLabelMin ~min-max-pp))
         :else
         (.getLabelMin min-max-pp)))

(defn get-max
  [min-max-pp & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [min-max-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getMax ~min-max-pp))
         :else
         (.getMax min-max-pp)))

(defn get-min
  [min-max-pp & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [min-max-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getMin ~min-max-pp))
         :else
         (.getMin min-max-pp)))

(defn get-target-max
  [min-max-pp & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [min-max-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getTargetMax ~min-max-pp))
         :else
         (.getTargetMax min-max-pp)))

(defn get-target-min
  [min-max-pp & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [min-max-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getTargetMin ~min-max-pp))
         :else
         (.getTargetMin min-max-pp)))

(defn load-min-max
  "Load the given min and max form the supplied file(s)

  :files (coll), collection of file paths to be loaded

  :pp (pre-processor). can be the standardizer or the min-max"
  [& {:keys [pp files as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :files (:or (_ :guard coll?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.load ~pp (array-of :data (map clojure.java.io/as-file ~files)
                                :java-type java.io.File)))
         :else
         (.load pp (array-of :data (map clojure.java.io/as-file files)
                             :java-type java.io.File))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; standardize specific fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-label-std
  [standardize-pp & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [standardize-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLabelStd ~standardize-pp))
         :else
         (.getLabelStd standardize-pp)))

(defn get-mean
  [standardize-pp & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [standardize-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getMean ~standardize-pp))
         :else
         (.getMean standardize-pp)))

(defn get-std
  [standardize-pp & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [standardize-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getStd ~standardize-pp))
         :else
         (.getStd standardize-pp)))

(defn get-label-mean
  [standardize-pp & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [standardize-pp]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLabelMean ~standardize-pp))
         :else
         (.getLabelMean standardize-pp)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vgg16 specific fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn preprocess-features!
  "preprcess the features

  :features (vec or INDArray), the features

  returns the processed features"
  [& {:keys [vgg16-pp features as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:vgg16-99 (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(.preProcess ~vgg16-pp (vec-or-matrix->indarray ~features)))
         :else
         (.preProcess vgg16-pp (vec-or-matrix->indarray features))))
