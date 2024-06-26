# async def read_file_as_image(file: UploadFile):
#     try:
#         if file.content_type in ["image/jpeg", "image/png"]:
#             img_byte = await file.read()
#
#             pil_img = Image.open(BytesIO(img_byte))  # PIL Image
#
#             resized_image = pil_img.resize(NEW_SIZE)
#             pil_img = pil_img.resize(IMAGENET_IMAGE_SIZE)
#
#             np_image: np.ndarray = np.array(resized_image)
#             img_shape: tuple = np_image.shape
#             if len(img_shape) == 3:  # color image
#                 if img_shape[2] == 4:  # RGBA
#                     logging.info("Input Image: RGBA")
#                     np_image = np_image[:, :, :3]
#                     np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
#                     return pil_img, np_image
#                 else:
#                     logging.info(f"Input Image: RGB")
#                     np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
#                     return pil_img, np_image
#             else:  # Gray image
#                 logging.info("Input Image: Gray")
#                 return pil_img, np.stack((np_image,) * 3, axis=-1)
#
#         else:
#
#             # unidentified file type and we do not process it
#             # such as pdf, GIF, BMP image, video, etc. mention in the log
#             print(f"INFO [Unsupported file type: {file.content_type}]")
#             return None, None
#
#     except Exception as e:
#         logging.error(f"Image byte to array conversion failure: {e}")
#         return None, None