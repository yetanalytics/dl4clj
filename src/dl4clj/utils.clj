(ns dl4clj.utils)

;; move contains-many? here and change all of the breaking changes that ensue
(defn contains-many? [m & ks]
  (every? #(contains? m %) ks))

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
  "ensures types are correct before being passed to java methods.  If they are not, throws
   -- I'd rather have the clojure compiler throw than the JVM
   -- these will eventually turn into clojure specs once 1.9 is officaly released"
  typez)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; standard types
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod type-checking :string [opts]
  (let [data (:string opts)]
    (if (= java.lang.String (type data))
      data
     (assert false "incorrect input type, should be a java.lang.String"))))

(defmethod type-checking :integer [opts]
  (let [data (:integer opts)]
    (if (= java.lang.Integer (type data))
      data
      (assert false "incorrect input type, should be an integer"))))

(defmethod type-checking :double [opts]
  (let [data (:double opts)]
    (if (= java.lang.Double (type data))
      data
      (assert false "incorrect input type, should be a double"))))

(defmethod type-checking :long [opts]
  (let [data (:long opts)]
    (if (= java.lang.Long (type data))
      data
      (assert false "incorrect input type, should be a long"))))

(defmethod type-checking :boolean [opts]
  (let [data (:boolean opts)]
    (if (= java.lang.Boolean (type data))
      data
      (assert false "incorrect input type, should be a boolean"))))

(defmethod type-checking :keyword [opts]
  (let [data (:keyword opts)]
    (if (= clojure.lang.Keyword (type data))
      data
      (assert false "incorrect input type, should be a keyword"))))

(defmethod type-checking :clojure-map [opts]
  (let [data (:clojure-map opts)]
    (if (= clojure.lang.PersistentArrayMap (type data))
      data
      (assert false "incorrect input type, should be a clojure map"))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dl4j types
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; add in all layer types
;; add in clustering types
(defmethod type-checking :INDArray [opts]
  (let [data (:INDArray opts)]
    (if (= org.nd4j.linalg.cpu.nativecpu.NDArray (type data))
      ;; or .gpu.NDArray
      data
      (assert false "incorrect input type, should be an NDArray"))))

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

(defmethod type-checking :evaluation [opts]
  (let [data (:evaluation opts)]
    (if (= org.deeplearning4j.eval.Evaluation
           (type data))
      data
      (assert false "incorrect input type, should be an Evaluation (classifcation)"))))

(defmethod type-checking :regression-evaluation [opts]
  (let [data (:regression-evaluation opts)]
    (if (= org.deeplearning4j.eval.RegressionEvaluation
           (type data))
      data
      (assert false "incorrect input type, should be an Evaluation (regression)"))))

(defmethod type-checking :default-step-fn [opts]
  (let [data (:default-step-fn opts)]
    (if (= org.deeplearning4j.optimize.stepfunctions.DefaultStepFunction
           (type data))
      data
      (assert false "incorrect input type, should be a default step function"))))

(defmethod type-checking :distribution [opts]
  (let [data (:distribution opts)
        dt (type data)]
    (if (or (= org.deeplearning4j.nn.conf.distribution.UniformDistribution dt)
            (= org.deeplearning4j.nn.conf.distribution.NormalDistribution dt)
            (= org.deeplearning4j.nn.conf.distribution.BinomialDistribution dt))
      data
      (assert false "incorrect input type, should be one of the possible distributions (uniform, normal, binomial)"))))

(defmethod type-checking :pre-processor [opts]
  (let [data (:pre-processor opts)
        dt (type data)]
    (if (or (= org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.UnitVarianceProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.ZeroMeanAndUnitVariancePreProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.ZeroMeanPrePreProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor dt)
            (= org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor dt))
      data
      (assert false "incorrect input type, should be one of the possible pre-processors"))))

(defmethod type-checking :default-gradient [opts]
  (let [data (:default-gradient opts)]
    (if (= org.deeplearning4j.nn.gradient.DefaultGradient (type data))
      data
      (assert false "incorrect input type, should be a default gradient"))))

(defmethod type-checking :record-reader [opts]
  ;; update this once other record readers are implemented
  (let [data (:record-reader opts)
        dt (type data)]
    (if (or (= org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader dt)
            (= org.datavec.api.records.reader.impl.csv.CSVRecordReader dt)
            (= org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader dt)
            (= org.datavec.api.records.reader.impl.FileRecordReader dt)
            (= org.datavec.api.records.reader.impl.LineRecordReader dt)
            (= org.datavec.api.records.reader.impl.collection.ListStringRecordReader dt))
      data
      (assert false "incorrect input type, should be one of the possible record readers"))))

(defmethod type-checking :file-split [opts]
  (let [data (:file-split opts)]
    (if (= org.datavec.api.split.FileSplit (type data))
      data
      (assert false "incorrect input type, should be a file split"))))

;; add in DataSetIterator after refactoring the ns

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dl4j enums
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod type-checking :activation-fn [opts]
  (let [data (:activation-fn opts)]
    (if (= org.nd4j.linalg.activations.Activation (type data))
      data
      (assert false "incorrect input type, should be an activation fn enum"))))

(defmethod type-checking :gradient-normalization [opts]
  (let [data (:gradient-normalization opts)]
    (if (= org.deeplearning4j.nn.conf.GradientNormalization (type data))
      data
      (assert false "incorrect input type, should be a gradient normalization enum"))))

(defmethod type-checking :learning-rate-policy [opts]
  (let [data (:learning-rate-policy opts)]
    (if (= org.deeplearning4j.nn.conf.LearningRatePolicy (type data))
      data
      (assert false "incorrect input type, should be a learning rate policy enum"))))

(defmethod type-checking :updater [opts]
  (let [data (:updater opts)]
    (if (= org.deeplearning4j.nn.conf.Updater (type data))
      data
      (assert false "incorrect input type, should be a updater enum"))))

(defmethod type-checking :weight-init [opts]
  (let [data (:weight-init opts)]
    (if (= org.deeplearning4j.nn.weights.WeightInit (type data))
      data
      (assert false "incorrect input type, should be a weight init enum"))))

(defmethod type-checking :loss-fn [opts]
  (let [data (:loss-fn opts)]
    (if (= org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction (type data))
      data
      (assert false "incorrect input type, should be a loss function enum"))))

(defmethod type-checking :hidden-unit [opts]
  (let [data (:hidden-unit opts)]
    (if (= org.deeplearning4j.nn.conf.layers.RBM$HiddenUnit (type data))
      data
      (assert false "incorrect input type, should be a hidden unit enum"))))

(defmethod type-checking :visible-unit [opts]
  (let [data (:visible-unit opts)]
    (if (= org.deeplearning4j.nn.conf.layers.RBM$VisibleUnit (type data))
      data
      (assert false "incorrect input type, should be a visible unit enum"))))

(defmethod type-checking :convolution-mode [opts]
  (let [data (:convolution-mode opts)]
    (if (= org.deeplearning4j.nn.conf.ConvolutionMode (type data))
      data
      (assert false "incorrect input type, should be a convolution mode enum"))))

(defmethod type-checking :cudnn-algo-mode [opts]
  (let [data (:cudnn-algo-mode opts)]
    (if (= org.deeplearning4j.nn.conf.layers.ConvolutionLayer$AlgoMode (type data))
      data
      (assert false "incorrect input type, should be a cudnn-algo-mode enum"))))

(defmethod type-checking :pooling-type [opts]
  (let [data (:pooling-type opts)]
    (if (= org.deeplearning4j.nn.conf.layers.PoolingType (type data))
      data
      (assert false "incorrect input type, should be a pooling type enum"))))

(defmethod type-checking :backprop-type [opts]
  (let [data (:backprop-type opts)]
    (if (= org.deeplearning4j.nn.conf.BackpropType (type data))
      data
      (assert false "incorrect input type, should be a backprop-type enum"))))

(defmethod type-checking :optimization-algorithm [opts]
  (let [data (:optimization-algorithm opts)]
    (if (= org.deeplearning4j.nn.api.OptimizationAlgorithm (type data))
      data
      (assert false "incorrect input type, should be an optimization algorithm enum"))))

(defmethod type-checking :mask-state [opts]
  (let [data (:mask-state opts)]
    (if (= org.deeplearning4j.nn.api.MaskState (type data))
      data
      (assert false "incorrect input type, should be a mask state enum"))))

(defmethod type-checking :layer-type [opts]
  (let [data (:layer-type opts)]
    (if (= org.deeplearning4j.nn.api.Layer$Type (type data))
      data
      (assert false "incorrect input type, should be a layer type enum"))))

(defmethod type-checking :layer-training-mode [opts]
  (let [data (:layer-training-mode opts)]
    (if (= org.deeplearning4j.nn.api.Layer$TrainingMode (type data))
      data
      (assert false "incorrect input type, should be a training mode enum"))))

(defmethod type-checking :input-type [opts]
  (let [data (:input-type opts)
        dt (type data)]
    (if (or (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeRecurrent dt)
            (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeFeedForward dt)
            (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeConvolutional dt)
            (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeConvolutionalFlat dt))
      data
      (assert false "incorrect input type, should be one of the layer input type enums"))))


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
