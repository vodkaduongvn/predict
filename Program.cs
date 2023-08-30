using BERTTokenizers;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;

namespace ConsoleApp2
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //var contentPdf = GetContentPDF(@"C:\WORK\M3\bert-csharp\Z000000002.pdf"); // pay slip
            //var contentPdf = GetContentPDF(@"C:\WORK\M3\bert-csharp\Z000000059.pdf"); // void check
            var contentPdf = GetContentPDF(@"C:\WORK\M3\bert-csharp\Z000000998.pdf"); // other document
            Console.WriteLine(contentPdf);

            // Create Tokenizer and tokenize the sentence.
            int maxSequenceLength = 512;
            var tokenizer = new BertUncasedLargeTokenizer();

            try
            {
                #region Tokenize the content pdf

                // Get the content tokens.
                var tokens = tokenizer.Tokenize(contentPdf);
                // Console.WriteLine(String.Join(", ", tokens));

                // Encode the sentence and pass in the count of the tokens in the sentence.
                var encoded = tokenizer.Encode(tokens.Count(), contentPdf);

                int inputLength = Math.Min(encoded.Count, maxSequenceLength);

                // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
                var bertInput = new BertInput()
                {
                    InputIds = encoded.Select(t => t.Item1).Take(inputLength).ToArray(),
                    AttentionMask = encoded.Select(t => t.Item3).Take(inputLength).ToArray(), 
                    TypeIds = encoded.Select(t => t.Item2).Take(inputLength).ToArray(),
                };

                // Create input tensor.
                var input_ids = ConvertToTensor(bertInput.InputIds, bertInput.InputIds.Length);
                var attention_mask = ConvertToTensor(bertInput.AttentionMask, bertInput.InputIds.Length);
                var token_type_ids = ConvertToTensor(bertInput.TypeIds, bertInput.InputIds.Length);

                #endregion

                // Get path to model to create inference session.
                var modelPath = @"C:\WORK\M3\bert-csharp\model.onnx";

                #region Do the prediction

                // Create an InferenceSession from the Model Path.
                var session = new InferenceSession(modelPath);

                // Run session and send the input data in to get inference output. 
                using (var results = session.Run(new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", input_ids),
                    NamedOnnxValue.CreateFromTensor("attention_mask", attention_mask),
                    NamedOnnxValue.CreateFromTensor("token_type_ids", token_type_ids)
                }))
                {
                    var logits = results.First().AsTensor<float>().ToArray();
                    var prediction = Array.IndexOf(logits, logits.Max());

                    var predictedLabel = GetPredictedLabel(prediction); // Provide a mapping from prediction to labels
                    Console.WriteLine($"\nPredicted Label: {predictedLabel}"); 
                }

                #endregion

            }
            catch (Exception ex)
            {

            }
        }

        public static Tensor<long> ConvertToTensor(long[] inputArray, int inputDimension)
        {
            // Create a tensor with the shape the model is expecting. Here we are sending in 1 batch with the inputDimension as the amount of tokens.
            Tensor<long> input = new DenseTensor<long>(new[] { 1, inputDimension });

            // Loop through the inputArray (InputIds, AttentionMask and TypeIds)
            for (var i = 0; i < inputArray.Length; i++)
            {
                // Add each to the input Tenor result.
                // Set index and array value of each input Tensor.
                input[0, i] = inputArray[i];
            }
            return input;
        }

        private static string GetContentPDF(string pdfFilePath)
        {
            //string pdfFilePath = "path_to_your_pdf_file.pdf";
            string text = "";
            try
            {
                using (PdfDocument pdfDocument = new PdfDocument(new PdfReader(pdfFilePath)))
                {
                    for (int pageNum = 1; pageNum <= pdfDocument.GetNumberOfPages(); pageNum++)
                    {
                        PdfPage page = pdfDocument.GetPage(pageNum);
                        ITextExtractionStrategy strategy = new SimpleTextExtractionStrategy();
                        text = PdfTextExtractor.GetTextFromPage(page, strategy);
                        text = Encoding.UTF8.GetString(Encoding.Convert(Encoding.ASCII, Encoding.UTF8, Encoding.Default.GetBytes(text)));
                        var formattedText = text.Replace("\n", "");
                        text = formattedText;
                        Console.WriteLine($"Page {pageNum}:\n{text}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }

            return text;
        }

        static string GetPredictedLabel(int prediction)
        {
            // Define your labels mapping here
            var labelsMapping = new Dictionary<int, string>
        {
            // Provide mapping from prediction value to labels
            { 0, "Letter from employer" },
            { 1, "Pay slip" },
            { 2, "T4 slips for the past 2 years" },
            { 3, "Proof of personal savings downpayment" },
            { 4, "Void Check" },
            { 5, "Offer of purchase" },
            { 6, "MLS Listing" },
            { 7, "Document 801" },
            { 8, "Document 802" },
            { 9, "Other document" },
        };

            return labelsMapping[prediction];
        }
    }

    public class BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public long[] TypeIds { get; set; }
    }
}