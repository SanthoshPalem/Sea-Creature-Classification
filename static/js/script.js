document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.querySelector("input[type='file']");
    const previewImage = document.createElement("img");
    previewImage.style.maxWidth = "100%";
    previewImage.style.marginTop = "10px";

    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                document.querySelector(".container").appendChild(previewImage);
            };
            reader.readAsDataURL(file);
        }
    });
});