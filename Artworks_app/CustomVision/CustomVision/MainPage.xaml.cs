using Newtonsoft.Json;
using Plugin.Media;
using Plugin.Media.Abstractions;
using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using Xamarin.Forms;

namespace CustomVision
{
    public partial class MainPage : ContentPage
    {
        public const string ServiceApiUrl = "https://artvisionlab.azurewebsites.net/predict";
        private MediaFile _foto = null;
        public MainPage()
        {
            InitializeComponent();
        }

        private async void ElegirImage(object sender, EventArgs e)
        {
            await CrossMedia.Current.Initialize();

            _foto = await Plugin.Media.CrossMedia.Current.PickPhotoAsync(new PickMediaOptions());
            Img.Source = FileImageSource.FromFile(_foto.Path);
        }

        private async void TomarFoto(object sender, EventArgs e)
        {
            await CrossMedia.Current.Initialize();

            if (!CrossMedia.Current.IsCameraAvailable || !CrossMedia.Current.IsTakePhotoSupported)
            {
                return;
            }

            var foto = await CrossMedia.Current.TakePhotoAsync(new StoreCameraMediaOptions()
            {
                PhotoSize = PhotoSize.Custom,
                CustomPhotoSize = 10,
                CompressionQuality = 92,
                Name = "image.jpg"
            });

            _foto = foto;

            if (_foto == null)
                return;

            Img.Source = FileImageSource.FromFile(_foto.Path);
        }

        private async void Clasificar(object sender, EventArgs e)
        {
            using (Acr.UserDialogs.UserDialogs.Instance.Loading("Clasificando..."))
            {
                if (_foto == null) return;

                var httpClient = new HttpClient();
                var url = ServiceApiUrl;
                var requestContent = new MultipartFormDataContent();
                var content = new StreamContent(_foto.GetStream());

                content.Headers.ContentType =
                    MediaTypeHeaderValue.Parse("image/jpg");

                requestContent.Add(content, "file", "image.jpg");

                var response = await httpClient.PostAsync(url, requestContent);

                if (!response.IsSuccessStatusCode)
                {
                    Acr.UserDialogs.UserDialogs.Instance.Toast("Hubo un error en la deteccion...");
                    return;
                }

                var json = await response.Content.ReadAsStringAsync();

                var prediction = JsonConvert.DeserializeObject<string>(json);
                if (prediction == null)
                {
                    Acr.UserDialogs.UserDialogs.Instance.Toast("Image no reconocida.");
                    return;
                }
                ResponseLabel.Text = $"{prediction}";
                //Accuracy.Progress = p.Probability;
            }

            Acr.UserDialogs.UserDialogs.Instance.Toast("Clasificacion terminada...");
        }
    }

    public class ClasificationResponse
    {     
        public string Tag { get; set; }
    }
}
