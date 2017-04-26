(ns dl4clj.utils
  (:import [org.deeplearning4j.nn.api Model Layer]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.deeplearning4j.datasets DataSets]
           [org.deeplearning4j.datasets.datavec
            RecordReaderDataSetIterator
            RecordReaderMultiDataSetIterator$Builder
            RecordReaderMultiDataSetIterator
            SequenceRecordReaderDataSetIterator
            SequenceRecordReaderDataSetIterator$AlignmentMode]))

;; move contains-many? here and change all of the breaking changes that ensue

(defn camelize
  "Turn a symbol or keyword or string to a camel-case verion, e.g. (camelize :foo-bar) => :FooBar"
  [x & capitalize?]
  (let [parts (clojure.string/split (name x) #"[\s_-]+")
        not-capitalized (clojure.string/join "" (cons (first parts)
                                                      (map #(str (clojure.string/upper-case (subs % 0 1)) (subs % 1))
                                                           (rest parts))))
        new-name (if capitalize?
                   (clojure.string/join [(clojure.string/upper-case (subs not-capitalized 0 1))
                                         (subs not-capitalized 1)])
                   not-capitalized)]
    (condp = (type x)
      java.lang.String new-name
      clojure.lang.Keyword (keyword new-name)
      clojure.lang.Symbol (symbol new-name))))

(defn camel-to-dashed
  "Turn a symbol or keyword or string like 'bigBlueCar' to 'big-blue-car'."
  [x & capitalize?]
  (let [parts (or (re-seq #"[a-xA-Z][A-Z\s_]*[^A-Z\s_]*" (name x))
                  [(name x)])
        new-name (clojure.string/join "-" (map clojure.string/lower-case parts))]
    (condp = (type x)
      java.lang.String new-name
      clojure.lang.Keyword (keyword new-name)
      clojure.lang.Symbol (symbol new-name))))

(defn indexed [col]
  (map vector col (range)))

(defn typez
  [opts]
  (first (keys opts)))

(defmulti type-checking
  "ensures types are correct before being passed to java methods.  If they are not, throws"
  typez)

(defmethod type-checking :INDArray [opts]
  (let [data (:INDArray opts)]
    (if (= org.nd4j.linalg.cpu.nativecpu.NDArray (type data))
      ;; or .gup.NDArray
      data
      (assert false "incorrect input type, should be an NDArray"))))

(defmethod type-checking :string [opts]
  (let [data (:string opts)]
    (if (= java.lang.String (type data))
      data
      (assert false "incorrect input type, should be a java.lang.String"))))

(defmethod type-checking :dataset [opts]
  (let [data (:dataset opts)]
    (if (= org.nd4j.linalg.dataset.DataSet (type data))
      data
      (assert false "incorrect input type, should be a DataSet"))))

(defmethod type-checking :alignment-mode [opts]
  (let [data (:alignment-mode opts)]
    (if (= org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator$AlignmentMode
           (type data))
      data
      (assert false "incorrect input type, should be an AlignmentMode enum"))))

;; Layer
;; gradient
;; nn's
;; model
;; NeuralNetConfiguration
;; IterationListener
;; Layer.TrainingMode
;; MaskState
;; array
;; collection
;; all the layer types
;; dataset Iterator
;; Updater
;; Tree
