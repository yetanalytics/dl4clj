(ns ^{:doc ""}
  dl4clj.examples.example-utils
  (:import [java.io IOException BufferedInputStream FileInputStream BufferedOutputStream FileOutputStream]
           [org.apache.commons.io FileUtils]
           [org.deeplearning4j.eval Evaluation]
           [org.apache.commons.compress.archivers.tar TarArchiveEntry TarArchiveInputStream]
           [org.apache.commons.compress.compressors.gzip GzipCompressorInputStream]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))

(defn init [^MultiLayerNetwork mln]
  (.init mln)
  mln)

(defn ex-train [^MultiLayerNetwork mln n-epoch iterator]
  (loop
      [i 0
       result {}]
    (cond (not= i n-epoch)
          (do
            (println "current at epoch:" i)
            (recur (inc i)
                   (.fit mln iterator)))
          (= i n-epoch)
          (do
            (println "training done")
            mln))))

(defn new-evaler [output-n]
  (Evaluation. output-n))

(defn eval-model [mln iterator evaler]
  (while (true? (.hasNext iterator))
    (let [nxt (.next iterator)
          output (.output mln (.getFeatureMatrix nxt))]
      (do (.eval evaler (.getLabels nxt) output)
          (println (.stats evaler))))))

(defn reset-iterator
  [t-iterator]
  (.reset t-iterator))



(defn index-map
  "Utility function to make an map from elements in a collection to indices"
  [col]
  (zipmap col (range)))

(def ^:const +minimal-character-set+
  (let [valid-chars (concat (for [c (range (int \a) (inc (int \z)))] (char c))
                            (for [c (range (int \A) (inc (int \Z)))] (char c))
                            (for [c (range (int \0) (inc (int \9)))] (char c))
                            [\!, \&, \(, \), \?, \-, \\, \", ,, \., \:, \;, \space , \newline, \tab])]
    (into #{} valid-chars)))

(def ^:const +default-character-set+
  (let [valid-chars (concat +minimal-character-set+
                            [\@, \#, \$, \%, \^, \*, \{, \}, \[, \], \/, \+, \_, \\\, \|, \<, \>])]
    (into #{} valid-chars)))

(defn shakespeare-file []
  "Downloads and returns the complete works of Shakespeare as a string.

   5.3MB file in UTF-8 Encoding, ~5.4 million characters
  https://www.gutenberg.org/ebooks/100
"
  (let [url "https://s3.amazonaws.com/dl4j-distribution/pg100.txt"
        temp-dir  (System/getProperty "java.io.tmpdir")
        file-location  (str temp-dir "/Shakespeare.txt")
        f (clojure.java.io/as-file file-location)]
    (when-not (.exists f)
      (do (FileUtils/copyURLToFile (clojure.java.io/as-url url) f)
          (println "File downloaded to " (.getAbsolutePath f))))
    (when-not (.exists f)
      (throw (IOException. (str "File does not exist: " file-location))))
    f))

(defn ^String shakespeare
  []
  (slurp (shakespeare-file)))

(defn extract-tgz
  ([file-path output-path]
   (extract-tgz file-path output-path {:buffer-size 4096}))
  ([^String file-path output-path {:keys [buffer-size]
                                   :or {buffer-size 4096}}]
   (println "Extracting " file-path "to" output-path)
   (with-open [is (TarArchiveInputStream. (GzipCompressorInputStream. (BufferedInputStream. (FileInputStream. file-path))))]
     (loop [entry (.getNextEntry is)]
       (when entry
         (if (.isDirectory entry)
           (.mkdirs (clojure.java.io/as-file (str output-path (.getName entry))))
           (let [data (bytes buffer-size)]
             (with-open [dest (BufferedOutputStream. (FileOutputStream. (str output-path (.getName entry))) buffer-size)]
               (loop [count (.read is data 0 buffer-size)]
                 (when-not (= -1 count)
                   (.write dest data 0 count)
                   (recur (.read is data 0 buffer-size))))))))))))
