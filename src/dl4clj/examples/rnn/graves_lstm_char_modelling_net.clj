(ns ^{:doc "
GravesLSTM Character modelling example
@author Joachim De Beule, based on Alex Black's java code, see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java
For general instructions using deeplearning4j's implementation of recurrent neural nets see http://deeplearning4j.org/usingrnns.html
"}
  dl4clj.examples.rnn.graves-lstm-char-modelling-net
  (:require [dl4clj.examples.rnn.character-iterator :refer :all])
  (:import [java.io File IOException]
           [java.net URL]
           [java.text CharacterIterator]
           [java.nio.charset Charset]
           [java.util Random]
           
           [org.deeplearning4j.datasets.iterator DataSetIterator]

           [org.deeplearning4j.nn.api Layer OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration Updater NeuralNetConfiguration NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf.distribution UniformDistribution]
           [org.deeplearning4j.nn.conf.layers GravesLSTM GravesLSTM$Builder RnnOutputLayer RnnOutputLayer$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]))

(defn- output-distribution 
  "Utility fn to convert a 1 dimensional NDArray to an array of doubles."
  [^INDArray output]
  (let [d (double-array (.length output))]
    (dotimes [i (.length output)]
      (aset d i (.getDouble output i)))
    d))

(defn- sample-from-distribution 
  "Given a probability distribution over discrete classes (an array of doubles), sample from the
  distribution and return the generated class index.
  @param distribution Probability distribution over classes. Must sum to 1.0.
"
  [^"[D" distribution] 
  (let [toss (rand)]
    (loop [i 0
           sum (aget distribution 0)]
      (cond (<= toss sum) i
            (< i (count distribution)) (recur (inc i) (+ sum (aget distribution i)))
            :else (throw (IllegalArgumentException. (str "Distribution is invalid? toss= " toss ", sum=" sum)))))))

(defn generate 
  "Generate a number of samples sample given an initialization string for 'priming' the RNN with a
  sequence to extend/continue."
  [^MultiLayerNetwork net initialization-string {:keys [valid-characters 
                                                        chars-per-sample 
                                                        num-samples]
                                                 :or {valid-characters +default-character-set+
                                                      num-samples 4
                                                      chars-per-sample 10}
                                                 :as opts}]
  (assert (not (empty? initialization-string)) "initialization string cannot be empty")
  (let [char-to-idx (index-map valid-characters)
        idx-to-char (zipmap (vals char-to-idx) (keys char-to-idx))
        initialization-input (Nd4j/zeros (int-array [num-samples (count valid-characters) (count initialization-string)]))
        sb (for [i (range num-samples)] (StringBuilder. ^String initialization-string))]
    
    ;; Fill input for initialization
    (dotimes [i (count initialization-string)]
      (let [idx (char-to-idx (nth initialization-string i))]
        (dotimes [j num-samples]
          (.putScalar ^INDArray initialization-input (int-array [j idx i]) (float 1.0)))))
    
    (.rnnClearPreviousState net)
    (loop [i 0 
           output (.tensorAlongDimension (.rnnTimeStep net initialization-input)
                                         (int (dec (count initialization-string)))
                                         (int-array [1 0]))]
      ;; Set up next input (single time step) by sampling from previous output
      (let [next-input (Nd4j/zeros (int num-samples) (int (count valid-characters)))]
        (dotimes [s num-samples]
          (let [sampled-character-idx (sample-from-distribution (output-distribution (.slice output s 1)))]
            (.putScalar next-input (int-array [s sampled-character-idx]) (float 1.0))
            (.append ^StringBuilder (nth sb s) (idx-to-char sampled-character-idx)))) ;; Add sampled character to StringBuilder (human readable output)
        (when (< i chars-per-sample)
          (recur (inc i)
                 (.rnnTimeStep net next-input)))))
    
    (map #(.toString ^StringBuilder %) sb)))

(defn graves-lstm-char-modelling-net 
  "Builds and returns an LSTM net."
  [{:keys [valid-characters
           lstm-layer-size ;; Number of units in each GravesLSTM layer
           learning-rate
           rms-decay
           iterations
           seed]
    :or {valid-characters +default-character-set+
         lstm-layer-size 200
         learning-rate 0.1
         rms-decay 0.95
         iterations 1
         seed 12345}
    :as opts}]
  (let [conf (-> (NeuralNetConfiguration$Builder.)
                 (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                 (.iterations (int iterations))
                 (.learningRate (double learning-rate))
                 (.rmsDecay (double rms-decay))
                 (.seed (int seed))
                 (.regularization true)
                 (.l2 0.001)
                 (.list (int 3))
                 (.layer 0 (-> (GravesLSTM$Builder.)
                               (.nIn (count valid-characters))
                               (.nOut (int lstm-layer-size))
                               (.updater Updater/RMSPROP)
                               (.activation "tanh")
                               (.weightInit WeightInit/DISTRIBUTION)
                               (.dist (UniformDistribution. -0.08 0.08))
                               (.build)))
                 (.layer 1  (-> (GravesLSTM$Builder.)
                                (.nIn (int lstm-layer-size))
                                (.nOut (int lstm-layer-size))
                                (.updater Updater/RMSPROP)
                                (.activation "tanh")
                                (.weightInit WeightInit/DISTRIBUTION)
                                (.dist (UniformDistribution. -0.08 0.08))
                                (.build)))
                 (.layer 2  (-> (RnnOutputLayer$Builder. (LossFunctions$LossFunction/MCXENT))
                                (.nIn (int lstm-layer-size))
                                (.nOut (count valid-characters))
                                (.activation "softmax")
                                (.updater Updater/RMSPROP)
                                (.weightInit WeightInit/DISTRIBUTION)
                                (.dist (UniformDistribution. -0.08 0.08))
                                (.build)))
                 (.pretrain false)
                 (.backprop true)
                 (.build))
        net (MultiLayerNetwork. conf)]
    (.init net)
    (.setListeners net [(ScoreIterationListener. (int 1))])
    
    ;; Print the  number of parameters in the network (and for each layer)
    (dotimes [i (count (.getLayers net))]
      (println "Number of parameters in layer "  i  ": "  (.numParams (nth (.getLayers net) i))))
    (println "Total number of network parameters: " (reduce + (map #(.numParams %) (.getLayers net))))
    
    net))

(defn train 
  "Performs a number of training epochs of an LSTM net on examples from a character-iterator. Prints
  generated samples in between epochs."
    [^MultiLayerNetwork net ^DataSetIterator character-dataset-iterator {:keys [epochs 
                                                                                generation-initialization-string 
                                                                                num-samples
                                                                                chars-per-sample
                                                                                valid-characters]
                                                                         :or {epochs 1
                                                                              num-samples 5
                                                                              chars-per-sample 50
                                                                              valid-characters +default-character-set+}
                                                                         :as opts}]
  (dotimes [i epochs]
    (.reset character-dataset-iterator)
    (.fit net character-dataset-iterator)
    (println "--------------------");
    (println "Completed epoch " i );
    (println (str "Sampling characters from network given initialization \"" generation-initialization-string "\""))
    (doseq [sample (generate net (or generation-initialization-string
                                     (str (rand-nth (seq valid-characters))))
                             opts)]
      (println "Sample: " sample)
      (println)))
  net)





(comment 

  ;;; Example usage:

  (def iter (shakespeare-iterator {:valid-characters +default-character-set+ 
                                   :batch-size 32
                                   :chars-per-segment 100
                                   :max-segments (* 32 50)}))

  (def net (graves-lstm-char-modelling-net {:valid-characters +default-character-set+ 
                                            :lstm-layer-size 200
                                            :iterations 1
                                            :learning-rate 0.1
                                            :rms-decay 0.95
                                            :seed 12345}))

  (train net iter {:valid-characters +default-character-set+
                   :epochs 30
                   :num-samples 4
                   :chars-per-sample 300})

  )

(comment 

  ;;; Example usage:

  (def iter (lr-character-iterator (subs (shakespeare) 0 1000) {}))

  (def net (graves-lstm-char-modelling-net {:valid-characters (:valid-chars iter)
                                            :lstm-layer-size 20
                                            :iterations 1
                                            :learning-rate 0.1
                                            :rms-decay 0.95
                                            :seed 12345}))

  (fit-and-sample net iter {:epochs 3
                            :num-samples 4
                            :chars-per-sample 100})

  (generate net "f" {})
  
  )

(defn random-algebraic-expression []
  (loop [ret "S"]
    (let [rule (rand-nth ["a" "b" "c" "x" "y" "z" "S+S" "S-S" "S/S" "S*S" "(S)"])
          parts (re-find #"(.*)S(.*)" ret)]
      (if parts 
        (recur (clojure.string/join [(nth parts 1) rule (nth parts 2)]))
        ret))))

(comment

  (random-algebraic-expression)



  (def NNCBuilder (NeuralNetConfiguration$Builder.))
  (instance? NeuralNetConfiguration$Builder NNCBuilder)
  (supers NeuralNetConfiguration$Builder)
  (parents NeuralNetConfiguration$Builder)
  (ancestors NeuralNetConfiguration$Builder)
  (bases NeuralNetConfiguration$Builder)

  (.iterations ^NeuralNetConfiguration$Builder NNCBuilder 1)


  (def NNCListBuilder (.list NNCBuilder 3))
  (instance? NeuralNetConfiguration$Builder NNCListBuilder)
  (instance? NeuralNetConfiguration$ListBuilder NNCListBuilder)
  (supers NeuralNetConfiguration$ListBuilder) ;; mlconf$builder
  (parents NeuralNetConfiguration$ListBuilder)
  (ancestors NeuralNetConfiguration$ListBuilder)
  (bases NeuralNetConfiguration$ListBuilder)


  (require '[dl4clj.nn.conf.updater :as updater])
  (require '[dl4clj.nn.api.optimization-algorithm :as opt])


  (def valid-characters +default-character-set+)
  (def lstm-layer-size 200)
  (def updater :rmsprop)
  (def optimization-algorithm :stochastic-gradient-descent)
  (def iterations 1)
  (def learning-rate 0.1)
  (def rms-decay 0.95)
  (def seed 12345)
  (-> (NeuralNetConfiguration$Builder.)
                 (.optimizationAlgo (opt/value-of optimization-algorithm))
                 (.iterations (int iterations))
                 (.learningRate (double learning-rate))
                 (.rmsDecay (double rms-decay))
                 (.seed (int seed))
                 (.regularization true)
                 (.l2 0.001)
                 (.list (int 3)) ;; makes this object a NeuralNetConfiguration$ListBuilder, which is an instance of MultiLayerConfiguration$Builder
                 (.layer 0 (-> (GravesLSTM$Builder.)
                               (.nIn (count valid-characters))
                               (.nOut (int lstm-layer-size))
                               (.updater (updater/value-of updater))
                               (.activation "tanh")
                               (.weightInit WeightInit/DISTRIBUTION)
                               (.dist (UniformDistribution. -0.01 0.01))
                               (.build)))
                 (.layer 1  (-> (GravesLSTM$Builder.)
                                (.nIn (int lstm-layer-size))
                                (.nOut (int lstm-layer-size))
                                (.updater (updater/value-of updater))
                                (.activation "tanh")
                                (.weightInit WeightInit/DISTRIBUTION)
                                (.dist (UniformDistribution. -0.01 0.01))
                                (.build)))
                 (.layer 2  (-> (RnnOutputLayer$Builder. (LossFunctions$LossFunction/MCXENT))
                                (.nIn (int lstm-layer-size))
                                (.nOut (count valid-characters))
                                (.activation "softmax")
                                (.updater (updater/value-of updater))
                                (.weightInit WeightInit/DISTRIBUTION)
                                (.dist (UniformDistribution. -0.01 0.01))
                                (.build)))
                 (.pretrain false)
                 (.backprop true))

  ;;; doesn't work:
  (-> ;; (MultiLayerConfiguration$Builder.) 
   (NeuralNetConfiguration$Builder.)
   (.optimizationAlgo (opt/value-of :stochastic-gradient-descent))
   (.iterations 1)
   (.learningRate 0.1)
   (.rmsDecay 0.95)
   (.seed 12345)
   (.regularization true)
   (.l2 0.001)
   (.list 3) ;; this makes a listbuilder!
   (.layer 0 (-> (GravesLSTM$Builder.)
                 (.nIn (count +default-character-set+))
                 (.nOut 200)
                 (.updater (updater/value-of :rmsprop))
                 (.activation "tanh")
                 (.weightInit WeightInit/DISTRIBUTION)
                 (.dist (UniformDistribution. -0.01 0.01))
                 (.build)))
   (.layer 1  (-> (GravesLSTM$Builder.)
                  (.nIn (int 200))
                  (.nOut (int 200))
                  (.updater (updater/value-of :rmsprop))
                  (.activation "tanh")
                  (.weightInit WeightInit/DISTRIBUTION)
                  (.dist (UniformDistribution. -0.01 0.01))
                  (.build)))
   (.layer 2  (-> (RnnOutputLayer$Builder. (LossFunctions$LossFunction/MCXENT))
                  (.nIn (int 200))
                  (.nOut (count +default-character-set+))
                  (.activation "softmax")
                  (.updater (updater/value-of :rmsprop))
                  (.weightInit WeightInit/DISTRIBUTION)
                  (.dist (UniformDistribution. -0.01 0.01))
                  (.build)))
   (.pretrain false)
   (.backprop true))


  
  (let [opt {:optimization-algo :stochastic-gradient-descent
             :num-iterations 1
             :learning-rate 0.1
             :rmsDecay 0.95
             :seed 12345
             :regularization true
             :l2 0.001
             :list 1
             :layers {0 (graves-lstm
                         {:n-in (count +default-character-set+)
                          :n-out 200
                          :updater :rmsprop
                          :activation "tanh"
                          :weight-init :distribution
                          :dist (uniform-distribution -0.01 0.01)})
                      ;; 1 (graves-lstm
                      ;;    {:nIn 200
                      ;;     :nOut 200
                      ;;     :updater :rmsprop
                      ;;     :activation "tanh"
                      ;;     :weight-init :distribution
                      ;;     :dist (uniform-distribution -0.01 0.01)})
                      ;; 2 (rnn-output-layer
                      ;;    {:loss-function :mcxent
                      ;;     :updater :rmsprop
                      ;;     :n-in 200
                      ;;     :n-out (count +default-character-set+)
                      ;;     :weight-init :distribution
                      ;;     :dist (uniform-distribution -0.01 0.01)})
                      }
             :pretrain false
             :backprop true}]
    (def cfg (neural-net-configuration opt)))

  )

      
      




;; (ns ^{:doc "

;; GravesLSTM Character modelling example

;; @author Joachim De Beule, based on Alex Black's java code, see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java

;; For general instructions using deeplearning4j's implementation of recurrent neural nets see http://deeplearning4j.org/usingrnns.html
;; "}
;;   dl4clj.examples.rnn.graves-lstm-char-modelling-net
;;   (:require [dl4clj.examples.rnn.lr-character-iterator :refer :all]
;;             [dl4clj.examples.example-utils :refer (+default-character-set+)]
;;             [dl4clj.nn.conf.layers.graves-lstm :refer (graves-lstm)]
;;             [dl4clj.nn.conf.layers.rnn-output-layer :refer (rnn-output-layer)]
;;             [dl4clj.nn.conf.distribution.uniform-distribution :refer (uniform-distribution)]
;;             [dl4clj.nn.conf.neural-net-configuration :refer (neural-net-configuration)])
;;   (:import [java.io IOException]
           
;;            [org.deeplearning4j.datasets.iterator DataSetIterator]
           
;;            [org.deeplearning4j.nn.conf MultiLayerConfiguration MultiLayerConfiguration$Builder BackpropType NeuralNetConfiguration NeuralNetConfiguration$Builder NeuralNetConfiguration$ListBuilder]
;;            [org.deeplearning4j.nn.conf.distribution UniformDistribution]
;;            [org.deeplearning4j.nn.conf.layers GravesLSTM GravesLSTM$Builder RnnOutputLayer RnnOutputLayer$Builder]
;;            [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
;;            [org.deeplearning4j.nn.weights WeightInit]
;;            [org.deeplearning4j.optimize.listeners ScoreIterationListener]


;;            ;; .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength().tBPTTBackwardLength()

;;            [org.nd4j.linalg.api.ndarray INDArray]
;;            [org.nd4j.linalg.factory Nd4j]
;;            [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
;;            [org.nd4j.linalg.dataset DataSet]))

;; (defn- output-distribution 
;;   "Utility fn to convert a 1 dimensional NDArray to an array of doubles."
;;   [^INDArray output]
;;   (let [d (double-array (.length output))]
;;     (dotimes [i (.length output)]
;;       (aset d i (.getDouble output i)))
;;     d))

;; (defn- sample-from-distribution 
;;   "Given a probability distribution over discrete classes (an array of doubles), sample from the
;;   distribution and return the generated class index.

;;   @param distribution Probability distribution over classes. Must sum to 1.0.
;; "
;;   [^"[D" distribution] 
;;   (let [toss (rand)]
;;     (loop [i 0
;;            sum (aget distribution 0)]
;;       (cond (<= toss sum) i
;;             (< i (count distribution)) (recur (inc i) (+ sum (aget distribution i)))
;;             :else (throw (IllegalArgumentException. (str "Distribution is invalid? toss= " toss ", sum=" sum)))))))

;; (defn sample
;;   "Generate a number of samples given an initialization string for 'priming' the RNN."
;;   [^MultiLayerNetwork net initialization-string char-to-idx {:keys [chars-per-sample 
;;                                                                     num-samples]
;;                                                              :or {num-samples 1
;;                                                                   chars-per-sample 100}
;;                                                              :as opts}]
;;   (assert (not (empty? initialization-string)) "initialization string cannot be empty")
;;   (let [idx-to-char (zipmap (vals char-to-idx)
;;                             (keys char-to-idx))
;;         initialization-input (Nd4j/zeros (int-array [num-samples (count char-to-idx) (count initialization-string)]))
;;         sb (for [i (range num-samples)] (StringBuilder. ^String initialization-string))]
    
;;     ;; Fill input for initialization
;;     (dotimes [i (count initialization-string)]
;;       (let [idx (char-to-idx (nth initialization-string i))]
;;         (dotimes [j num-samples]
;;           (.putScalar ^INDArray initialization-input (int-array [j idx i]) 1.0))))
    
;;     (.rnnClearPreviousState net)
;;     (loop [i 0 
;;            output (.tensorAlongDimension (.rnnTimeStep net initialization-input)
;;                                          (int (dec (count initialization-string)))
;;                                          (int-array [1 0]))]
;;       ;; Set up next input (single time step) by sampling from previous output
;;       (let [next-input (Nd4j/zeros (int num-samples) (int (count char-to-idx)))]
;;         (dotimes [s num-samples]
;;           (let [sampled-character-idx (sample-from-distribution (output-distribution (.slice output s 1)))]
;;             (.putScalar next-input (int-array [s sampled-character-idx]) (float 1.0))
;;             (.append ^StringBuilder (nth sb s) (idx-to-char sampled-character-idx)))) ;; Add sampled character to StringBuilder (human readable output)
;;         (when (< i chars-per-sample)
;;           (recur (inc i)
;;                  (.rnnTimeStep net next-input)))))
;;     (map #(.toString ^StringBuilder %) sb)))

;; (defn graves-lstm-char-modelling-net 
;;   "Builds and returns an LSTM net."
;;   [{:keys [valid-characters
;;            lstm-layer-size ;; Number of units in each GravesLSTM layer
;;            learning-rate
;;            rms-decay
;;            iterations
;;            seed
;;            pretrain
;;            use-tbptt-backprop
;;            tbptt-forward-length
;;            tbptt-backward-length
;;            updater
;;            optimization-algorithm]
;;     :or {updater :adagrad
;;          optimization-algorithm :stochastic-gradient-descent
;;          valid-characters +default-character-set+
;;          lstm-layer-size 200
;;          learning-rate 0.1
;;          rms-decay 0.95
;;          iterations 1
;;          seed 12345
;;          use-tbptt-backprop true
;;          pretrain false}
;;     :as opts}]
;;   (when use-tbptt-backprop
;;     (when-not tbptt-backward-length
;;       (throw (IllegalArgumentException. "missing required parameter tbptt-backward-length")))
;;     (when-not tbptt-forward-length
;;       (throw (IllegalArgumentException. "missing required parameter tbptt-forward-length")))
;;     (when (< tbptt-forward-length tbptt-backward-length)
;;       (throw (IllegalArgumentException. (str "tbptt-forward-length=" tbptt-forward-length " should not be smaller than tbptt-backward-length=" tbptt-backward-length))))             )
;;   (let [conf (-> (NeuralNetConfiguration$Builder.)
;;                  (.optimizationAlgo (opt/value-of optimization-algorithm))
;;                  (.iterations (int iterations))
;;                  (.learningRate (double learning-rate))
;;                  (.rmsDecay (double rms-decay))
;;                  (.seed (int seed))
;;                  (.regularization true)
;;                  (.l2 0.001)
;;                  (.list (int 3)) ;; makes this object a NeuralNetConfiguration$ListBuilder, which is an instance of MultiLayerConfiguration$Builder
;;                  (.layer 0 (-> (GravesLSTM$Builder.)
;;                                (.nIn (count valid-characters))
;;                                (.nOut (int lstm-layer-size))
;;                                (.updater (updater/value-of updater))
;;                                (.activation "tanh")
;;                                (.weightInit WeightInit/DISTRIBUTION)
;;                                (.dist (UniformDistribution. -0.01 0.01))
;;                                (.build)))
;;                  (.layer 1  (-> (GravesLSTM$Builder.)
;;                                 (.nIn (int lstm-layer-size))
;;                                 (.nOut (int lstm-layer-size))
;;                                 (.updater (updater/value-of updater))
;;                                 (.activation "tanh")
;;                                 (.weightInit WeightInit/DISTRIBUTION)
;;                                 (.dist (UniformDistribution. -0.01 0.01))
;;                                 (.build)))
;;                  (.layer 2  (-> (RnnOutputLayer$Builder. (LossFunctions$LossFunction/MCXENT))
;;                                 (.nIn (int lstm-layer-size))
;;                                 (.nOut (count valid-characters))
;;                                 (.activation "softmax")
;;                                 (.updater (updater/value-of updater))
;;                                 (.weightInit WeightInit/DISTRIBUTION)
;;                                 (.dist (UniformDistribution. -0.01 0.01))
;;                                 (.build)))
;;                  (.pretrain false)
;;                  (.backprop true))]
;;     (when use-tbptt-backprop 
;;       (.backpropType conf (BackpropType/TruncatedBPTT))
;;       (.tBPTTForwardLength conf tbptt-forward-length)
;;       (.tBPTTBackwardLength conf tbptt-backward-length))
;;     (let [net (MultiLayerNetwork. (.build conf))]
;;       (.init net)
;;       (.setListeners net [(ScoreIterationListener. (int 1))])
    
;;       ;; Print the  number of parameters in the network (and for each layer)
;;       (dotimes [i (count (.getLayers net))]
;;         (println "Number of parameters in layer "  i  ": "  (.numParams (nth (.getLayers net) i))))
;;       (println "Total number of network parameters: " (reduce + (map #(.numParams %) (.getLayers net))))
    
;;       net)))

;; (defn fit-and-sample
;;   "Performs a number of training epochs of an LSTM net on examples from a character-iterator. Prints
;;   generated samples in between epochs."
;;     [^MultiLayerNetwork net ^DataSetIterator iter {:keys [epochs 
;;                                                           generation-initialization-string 
;;                                                           num-samples
;;                                                           chars-per-sample]
;;                                                    :or {epochs 1
;;                                                         num-samples 5
;;                                                         chars-per-sample 100}
;;                                                    :as opts}]
;;     (loop [i 0]
;;       (println "--------------------")
;;       (println "Epoch" i)
;;       (.reset iter)
;;       (.rnnClearPreviousState net)
;;       (let [cnt (atom 0)]
;;         (doseq [^DataSet batch (take 10 (iterator-seq iter))]
;;           (.fit net batch)
;;           (when (zero? (mod cnt 100))
;;             (println (generate net (or generation-initialization-string
;;                                        (str (rand-nth (seq (:valid-chars iter)))))
;;                                opts))))
;;         (println "Completed epoch " i)
;;         (when (< i epochs) (recur (inc i))))))


