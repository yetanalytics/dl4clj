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
            [dl4clj.utils :refer [array-of obj-or-code? eval-if-code]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

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
         (let [[pp-obj ds-obj] (eval-if-code [pre-processor seq?] [ds seq?])]
           (do (.preProcess pp-obj ds-obj)
               ds-obj))))

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
         (let [[iter-obj ds-obj] (eval-if-code [iter seq?] [dataset seq?])]
           (doto (reset-iterator! iter-obj) (.preProcess ds-obj)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; generic normalizer (pre-processor) fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-pre-processor-type
  "Get the enum type of this pre-processor"
  [pre-processor & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [pre-processor]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getType ~pre-processor))
         :else
         (.getType pre-processor)))

(defn fit-iter!
  "Iterates over a dataset accumulating statistics for normalization

  returns the fit pre-processor"
  [& {:keys [pre-processor iter as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pre-processor (_ :guard seq?)
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~pre-processor (.fit ~iter)))
         :else
         (let [[norm-obj iter-obj] (eval-if-code [pre-processor seq?] [iter seq?])]
           (doto norm-obj (.fit iter-obj)))))

(defn fit-labels!?
  "Flag to specify if the labels/outputs in the dataset should be also normalized.

  returns the pre-processor"
  [& {:keys [pre-processor fit-labels? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pre-processor (_ :guard seq?)
           :fit-labels (:or (_ :guard boolean?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~pre-processor (.fitLabel ~fit-labels?)))
         :else
         (let [[norm-obj fit-labels-b] (eval-if-code [pre-processor seq?]
                                                     [fit-labels? seq? boolean?])]
           (doto norm-obj (.fitLabel fit-labels-b)))))

(defn normalize-labels?
  "Whether normalization for the labels is also enabled."
  [pre-processor & {:keys [as-code?]
                 :or {as-code? true}}]
  (match [pre-processor]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.isFitLabel ~pre-processor))
         :else
         (.isFitLabel pre-processor)))

(defn revert-features!
  "Undo the normalization applied by the pre-processor on the features array

  :features (vec or INDArray), nn input

  :features-mask (vec or INDArray), mask for the nn-input

  returns the (un)normalized features"
  [& {:keys [pre-processor features features-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pre-processor (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :features-mask (:or (_ :guard vector?)
                               (_ :guard seq?))}]
         (obj-or-code? as-code? `(.revertFeatures ~pre-processor (vec-or-matrix->indarray ~features)
                                                  (vec-or-matrix->indarray ~features-mask)))
         [{:pre-processor _
           :features _
           :features-mask _}]
         (let [[norm-obj features-vec mask-vec] (eval-if-code [pre-processor seq?]
                                                              [features seq?]
                                                              [features-mask seq?])]
           (.revertFeatures norm-obj (vec-or-matrix->indarray features-vec)
                            (vec-or-matrix->indarray mask-vec)))
         [{:pre-processor (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(.revertFeatures ~pre-processor (vec-or-matrix->indarray ~features)))
         [{:pre-processor _
           :features _}]
         (let [[norm-obj feature-vec] (eval-if-code [pre-processor seq?] [features seq?])]
           (.revertFeatures norm-obj (vec-or-matrix->indarray feature-vec)))))

(defn revert-labels!
  "Undo the normalization applied by the pre-processor on the labels array

  :labels (vec or INDArray), nn targets

  :labels-mask (vec or INDArray), mask for the nn-targets

  returns the (un)normalized labels"
  [& {:keys [pre-processor labels labels-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pre-processor (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.revertLabels ~pre-processor (vec-or-matrix->indarray ~labels)
                          (vec-or-matrix->indarray ~labels-mask)))
         [{:pre-processor _
           :labels _
           :labels-mask _}]
         (let [[norm-obj labels-vec mask-vec] (eval-if-code [pre-processor seq?]
                                                            [labels seq?]
                                                            [labels-mask seq?])]
           (.revertLabels norm-obj (vec-or-matrix->indarray labels-vec)
                          (vec-or-matrix->indarray mask-vec)))
         [{:pre-processor (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(.revertLabels ~pre-processor (vec-or-matrix->indarray ~labels)))
         [{:pre-processor _
           :labels _}]
         (let [[norm-obj labels-vec] (eval-if-code [pre-processor seq?] [labels seq?])]
           (.revertLabels norm-obj (vec-or-matrix->indarray labels-vec)))))

(defn transform-features!
  "applies the transform specified by the pre-processor to the features

  :features (vec or INDArray), nn input

  :features-mask (vec or INDArray), mask for the nn-input

  returns the normalized features"
  [& {:keys [pre-processor features features-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pre-processor (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :features-mask (:or (_ :guard vector?)
                               (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transform ~pre-processor (vec-or-matrix->indarray ~features)
                       (vec-or-matrix->indarray ~features-mask)))
         [{:pre-processor _
           :features _
           :features-mask _}]
         (let [[norm-obj features-vec mask-vec] (eval-if-code [pre-processor seq?]
                                                              [features seq?]
                                                              [features-mask seq?])]
           (.transform norm-obj (vec-or-matrix->indarray features-vec)
                       (vec-or-matrix->indarray mask-vec)))
         [{:pre-processor (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transform ~pre-processor (vec-or-matrix->indarray ~features)))
         [{:pre-processor _
           :features _}]
         (let [[norm-obj features-vec] (eval-if-code [pre-processor seq?]
                                                     [features seq?])]
           (.transform norm-obj (vec-or-matrix->indarray features-vec)))))

(defn transform-labels!
  "applies the transform specified by the pre-processor to the labels

  :labels (vec or INDArray), nn targets

  :labels-mask (vec or INDArray), mask for the nn-targets

  returns the normalized labels"
  [& {:keys [pre-processor labels labels-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pre-processor (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transformLabel ~pre-processor (vec-or-matrix->indarray ~labels)
                            (vec-or-matrix->indarray ~labels-mask)))
         [{:pre-processor _
           :labels _
           :labels-mask _}]
         (let [[norm-obj labels-vec mask-vec] (eval-if-code [pre-processor seq?]
                                                            [labels seq?]
                                                            [labels-mask seq?])]
           (.transformLabel norm-obj (vec-or-matrix->indarray labels-vec)
                            (vec-or-matrix->indarray mask-vec)))
         [{:pre-processor (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.transformLabel ~pre-processor (vec-or-matrix->indarray ~labels)))
         [{:pre-processor _
           :labels _}]
         (let [[norm-obj labels-vec] (eval-if-code [pre-processor seq?] [labels seq?])]
           (.transformLabel norm-obj (vec-or-matrix->indarray labels-vec)))))

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
         (let [[pp-obj features-vec] (eval-if-code [vgg16-pp seq?] [features seq?])]
           (.preProcess pp-obj (vec-or-matrix->indarray features-vec)))))
