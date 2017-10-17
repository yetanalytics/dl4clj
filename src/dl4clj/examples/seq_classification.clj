(ns dl4clj.examples.seq-classification
  (:require [dl4clj.datasets.iterators :as iter]
            [dl4clj.datasets.api.iterators :refer [set-pre-processor!
                                                   get-pre-processor
                                                   reset-iter!
                                                   next-example!]]
            [dl4clj.datasets.record-readers :as rr]
            [dl4clj.datasets.api.record-readers :refer [initialize-rr!]]
            [dl4clj.datasets.input-splits :as is]
            [dl4clj.datasets.pre-processors :refer [new-standardize-normalization-ds-preprocessor]]
            [dl4clj.datasets.api.pre-processors :refer [fit-iter!]]
            [dl4clj.nn.conf.builders.nn :as nn-conf]
            [dl4clj.nn.conf.builders.layers :as l]
            [dl4clj.nn.multilayer.multi-layer-network :as mln]
            [dl4clj.nn.api.multi-layer-network :refer [evaluate-classification output]]
            [dl4clj.nn.api.model :refer [init! set-listeners!]]
            [dl4clj.optimize.listeners :as listener]
            [dl4clj.utils :as u]
            [dl4clj.eval.api.eval :refer [get-stats get-prediction-by-predicted-class]])
  (:import [org.apache.commons.io IOUtils]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; notes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; implement the lstm regression example before the commit in which it was removed
;; becomes burried

;; set to 3 to test this works.  set back to 40 once satisified
;; need to change train-mln-with-ds-iter to return code
;; currently very ineffecient, creates a single object multiple times
;; need to have defined flows which account for this
;; use spark training as an example

;; there is some touching up I can do for this
;; but hit around 95 % accuracy
;; use this example to create more effecient flows given types of problems
;; This can be the variable length seq classification example
;; will need a way to make it more general
;; for the most part it is, its just the downloading and splitting
;; of the dataset and the alignment mode which is unqiue

;; need consistent lang for optimization algo, both optimization algo and optimization algorithm show up
;; no way to determine which one to use without checking fn

;; talking about set-pre-processor
;; I think the best way of handling combo flows like this
;; is to return a map with the keys being the type of objects
;; and the values being the code that evaluates to them
;; so users dont lose access to certain objects by using the flow

;; run into the issue that I now can easily normalize my testing iter
;; have to get the pre-processor from the iter
;; see get-pre-processor call bellow

(comment
  ;; saves 7 milsec due to the single object being passed down
  ;; validation for setting up predefined flows
  (time (eval (set-pre-processor :iter train-iter
                                 :normalizer (new-standardize-normalization-ds-preprocessor))))


  (time (eval (set-pre-processor! :pre-processor
                                  (fit-iter! :normalizer (new-standardize-normalization-ds-preprocessor)
                                             :iter train-iter)
                                  :iter train-iter))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; get the data
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def train-features-dir "dev-resources/uci/train/features/")

(def train-labels-dir "dev-resources/uci/train/labels/")

(def test-features-dir "dev-resources/uci/test/features/")

(def test-labels-dir "dev-resources/uci/test/labels/")

(defn collect-and-randomize-data
  [& {:keys [seed]}]
  (let [data (-> "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
                 clojure.java.io/as-url
                 IOUtils/toString
                 clojure.string/split-lines)

        transposed-data (for [each data]
                          (.replaceAll each " +" "\n"))]
    (loop [accum []
           source transposed-data
           counter 0]
      (if (empty? source)
        (if seed
          (u/shuffle* :coll accum :seed seed)
          (u/shuffle* :coll accum))
        (recur (conj accum  [(first source) (str (int (/ counter 100)))])
               (rest source)
               (inc counter))))))

(defn create-and-populate-file
  [& {:keys [data idx dir]}]
  (let [f-name (str dir idx ".csv")
        _ (clojure.java.io/make-parents f-name)]
    (spit f-name data)))

(defn populate-dirs
  [data]
  (if (.exists (clojure.java.io/as-file "dev-resources/uci"))
    (println "directory already exists")
    (loop [test {}
           train {}
           source data
           append-file-name 0]
      (if (empty? source)
        (println "done splitting data source")
        (let [cur-row (first source)
              [series label] cur-row]
          (if (< append-file-name 450)
            (recur (create-and-populate-file
                    :data series :idx append-file-name :dir train-features-dir)
                   (create-and-populate-file
                    :data label :idx append-file-name :dir train-labels-dir)
                   (rest source)
                   (inc append-file-name))
            (recur (create-and-populate-file
                    :data series :idx append-file-name :dir test-features-dir)
                   (create-and-populate-file
                    :data label :idx append-file-name :dir test-labels-dir)
                   (rest source)
                   (inc append-file-name))))))))

;; already populated
#_(populate-dirs (collect-and-randomize-data :seed 12345))

(defn get-abs-path
  [f-name]
  (.getAbsolutePath (clojure.java.io/as-file f-name)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dataset setup (training and testing)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def train-features-input-split (is/new-numbered-file-input-split
                                 :base-string (str (get-abs-path train-features-dir) "/" "%d.csv")
                                 :inclusive-min-idx 0
                                 :inclusive-max-idx 449))

(def train-labels-input-split (is/new-numbered-file-input-split
                               :base-string (str (get-abs-path train-labels-dir) "/" "%d.csv")
                               :inclusive-min-idx 0
                               :inclusive-max-idx 449))

(def train-features-seq-rr (initialize-rr! :rr (rr/new-csv-seq-record-reader)
                                           :input-split train-features-input-split))

(def train-labels-seq-rr (initialize-rr! :rr (rr/new-csv-seq-record-reader)
                                         :input-split train-labels-input-split))

(def train-iter (iter/new-seq-record-reader-dataset-iterator
                 :features-reader train-features-seq-rr
                 :labels-reader train-labels-seq-rr
                 :mini-batch-size 10
                 :n-possible-labels 6
                 :regression? false
                 :alignment-mode :align-end))

;; we are effecient up to creation of the iterator

(next-example! :iter (reset-iter! :iter train-iter))

(defn set-pre-processor
  [& {:keys [iter normalizer]}]
  (let [i (u/gensym* :sym "the-iter")
        norm (u/gensym* :sym "the-normalizer")]
    `(let [~i ~iter
           ~norm (fit-iter! :normalizer ~normalizer
                            :iter ~i)]
       (set-pre-processor! :pre-processor ~norm
                           :iter ~i))))

(def normalized-train-iter (set-pre-processor
                            :iter train-iter
                            :normalizer (new-standardize-normalization-ds-preprocessor)))


(def test-features-input-split (is/new-numbered-file-input-split
                                :base-string (str (get-abs-path test-features-dir) "/" "%d.csv")
                                :inclusive-min-idx 450
                                :inclusive-max-idx 599))

(def test-labels-input-split (is/new-numbered-file-input-split
                              :base-string (str (get-abs-path test-labels-dir) "/" "%d.csv")
                              :inclusive-min-idx 450
                              :inclusive-max-idx 599))


(def test-features-rr (initialize-rr! :rr (rr/new-csv-seq-record-reader)
                                      :input-split test-features-input-split))

(def test-labels-rr (initialize-rr! :rr (rr/new-csv-seq-record-reader)
                                    :input-split test-labels-input-split))

(def test-iter (iter/new-seq-record-reader-dataset-iterator
                :features-reader test-features-rr
                :labels-reader test-labels-rr
                :mini-batch-size 10
                :n-possible-labels 6
                :regression? false
                :alignment-mode :align-end))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; model config
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def mln-conf (nn-conf/builder
               :seed 123
               :optimization-algo :stochastic-gradient-descent
               :default-weight-init :xavier
               :default-updater :nesterovs
               :default-learning-rate 0.005
               :default-gradient-normalization :clip-element-wise-absolute-value
               :default-gradient-normalization-threshold 0.5
               :pretrain? false
               :backprop? true
               :layers {0 (l/graves-lstm-layer-builder :activation-fn :tanh
                                                       :n-in 1
                                                       :n-out 10)
                        1 (l/rnn-output-layer-builder :loss-fn :mcxent
                                                      :activation-fn :softmax
                                                      :n-in 10 :n-out 6)}))

(def model (set-listeners!
            :model (init! :model (mln/new-multi-layer-network :conf mln-conf))
            :listeners (listener/new-score-iteration-listener :print-every-n 20)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; train and evaluate
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def normalized-test-iter (set-pre-processor! :iter test-iter
                                              :pre-processor (get-pre-processor
                                                              :iter normalized-train-iter)
                                              :as-code? false))

(def trained-model (mln/train-mln-with-ds-iter! :mln model
                                                :iter normalized-train-iter
                                                :n-epochs 15
                                                :as-code? false))

(def evaler (evaluate-classification :mln trained-model
                                     :iter normalized-test-iter))

(clojure.pprint/pprint (get-stats :evaler evaler))
